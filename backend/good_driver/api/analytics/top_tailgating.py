from __future__ import annotations

import gzip
import heapq
import json
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

from .common import VIDEO_EXTENSIONS, _data_dir, _dist_for_second, _get_fps

router = APIRouter()

_MIN_SPEED = 80
_TAILGATING_WINDOWS = [
    (10, "10 seconds"),
    (30, "30 seconds"),
    (60, "1 minute"),
    (120, "2 minutes"),
    (300, "5 minutes"),
    (600, "10 minutes"),
]
_BIN_SIZE = 10


def _discover_speed_buckets(directory: str) -> list[int]:
    """Scan data to find all 10 km/h speed buckets >= _MIN_SPEED that have following distance."""
    data_dir = Path(directory)
    buckets: set[int] = set()
    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        ddir = _data_dir(data_dir, f.name)
        gps_path = ddir / "gps.json.gz"
        dist_path = ddir / "distances.json.gz"
        if not gps_path.exists() or not dist_path.exists():
            continue
        gps_data = json.loads(gzip.decompress(gps_path.read_bytes()))
        dist_data = json.loads(gzip.decompress(dist_path.read_bytes()))
        fps = _get_fps(ddir)
        for i in range(len(gps_data)):
            gps = gps_data[i]
            if gps is None or gps.get("speed_kmh") is None:
                continue
            speed = float(gps["speed_kmh"])
            if speed < _MIN_SPEED:
                continue
            dist_entry = _dist_for_second(dist_data, i, fps)
            if dist_entry is None or dist_entry.get("distance") is None:
                continue
            buckets.add(int(speed) // _BIN_SIZE * _BIN_SIZE)
    return sorted(buckets, reverse=True)


def _collect_tailgating_segments(
    directory: str, speed_bucket: int,
) -> list[list[dict]]:
    """Collect contiguous runs of seconds in a speed bucket with following distance.

    A point qualifies when GPS speed falls in [speed_bucket, speed_bucket + BIN_SIZE)
    and a following distance is available.  A new segment starts when a qualifying
    point is missing or there is a timestamp gap > 5 s.
    """
    data_dir = Path(directory)

    all_points: list[dict] = []
    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        ddir = _data_dir(data_dir, f.name)
        snap_path = ddir / "snap_to_road.json.gz"
        gps_path = ddir / "gps.json.gz"
        dist_path = ddir / "distances.json.gz"
        if not snap_path.exists() or not gps_path.exists() or not dist_path.exists():
            continue
        snap_data = json.loads(gzip.decompress(snap_path.read_bytes()))
        gps_data = json.loads(gzip.decompress(gps_path.read_bytes()))
        dist_data = json.loads(gzip.decompress(dist_path.read_bytes()))
        fps = _get_fps(ddir)
        for i, snap_entry in enumerate(snap_data):
            if snap_entry is None:
                continue
            ts_str = snap_entry.get("timestamp")
            if ts_str is None:
                continue
            gps = gps_data[i] if i < len(gps_data) else None
            if gps is None or gps.get("speed_kmh") is None:
                continue
            speed = float(gps["speed_kmh"])
            if not (speed_bucket <= speed < speed_bucket + _BIN_SIZE):
                continue
            dist_entry = _dist_for_second(dist_data, i, fps)
            if dist_entry is None or dist_entry.get("distance") is None:
                continue
            all_points.append({
                "ts": datetime.fromisoformat(ts_str.replace("Z", "+00:00")),
                "speed_kmh": speed,
                "distance": float(dist_entry["distance"]),
                "lat": snap_entry["lat"],
                "lon": snap_entry["lon"],
                "video": f.name,
                "second": i,
            })

    all_points.sort(key=lambda p: p["ts"])

    segments: list[list[dict]] = []
    current: list[dict] = []
    prev_ts: datetime | None = None

    for pt in all_points:
        gap_secs = (pt["ts"] - prev_ts).total_seconds() if prev_ts is not None else 0
        if gap_secs > 5:
            if current:
                segments.append(current)
                current = []
        current.append(pt)
        prev_ts = pt["ts"]

    if current:
        segments.append(current)

    return segments


def _find_closest_sections(
    segments: list[list[dict]],
    window_seconds: int,
    top_n: int = 20,
) -> list[dict]:
    """Sliding-window search for the top-N closest (lowest avg distance) sections."""
    # Max-heap of (-avg_distance, counter, info_dict) — we want lowest distance,
    # so we negate and use a max-heap (heapq is min-heap, negate twice).
    heap: list[tuple[float, int, dict]] = []
    counter = 0

    for entries in segments:
        n = len(entries)
        if n < window_seconds:
            continue

        running_sum = sum(e["distance"] for e in entries[:window_seconds])

        for start in range(n - window_seconds + 1):
            if start > 0:
                running_sum += entries[start + window_seconds - 1]["distance"]
                running_sum -= entries[start - 1]["distance"]

            avg = running_sum / window_seconds
            mid = entries[start + window_seconds // 2]
            start_entry = entries[start]
            info = {
                "mid_lat": mid["lat"],
                "mid_lon": mid["lon"],
                "mid_ts": mid["ts"].isoformat(),
                "start_lat": start_entry["lat"],
                "start_lon": start_entry["lon"],
                "video": start_entry["video"],
                "second": start_entry["second"],
                "avg_speed_kmh": round(
                    sum(e["speed_kmh"] for e in entries[start:start + window_seconds]) / window_seconds, 1,
                ),
            }

            if len(heap) < top_n:
                heapq.heappush(heap, (-avg, counter, info))
                counter += 1
            elif -avg > heap[0][0]:  # avg < current max distance in heap
                heapq.heapreplace(heap, (-avg, counter, info))
                counter += 1

    # Sort by avg distance ascending (closest first)
    results = sorted(heap, key=lambda t: t[0], reverse=True)
    return [
        {
            "avg_distance_m": round(-neg_avg, 1),
            "avg_speed_kmh": info["avg_speed_kmh"],
            "mid_lat": info["mid_lat"],
            "mid_lon": info["mid_lon"],
            "mid_ts": datetime.fromisoformat(info["mid_ts"]),
            "date": datetime.fromisoformat(info["mid_ts"]).astimezone().strftime("%Y-%m-%d %H:%M"),
            "start_lat": info["start_lat"],
            "start_lon": info["start_lon"],
            "video": info["video"],
            "second": info["second"],
        }
        for neg_avg, _, info in results
    ]


async def _reverse_geocode(
    client: httpx.AsyncClient, nominatim_url: str, lat: float, lon: float,
) -> tuple[str, str]:
    """Return (display_name, municipality) for given coordinates."""
    fallback = f"{lat:.5f}, {lon:.5f}"
    try:
        resp = await client.get(
            f"{nominatim_url.rstrip('/')}/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16,
                     "addressdetails": 1},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            display_name = data.get("display_name", fallback)
            address = data.get("address", {})
            municipality = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
                or address.get("county")
                or fallback
            )
            return display_name, municipality
    except Exception:
        pass
    return fallback, fallback


@router.get("/top-tailgating-sections")
async def top_tailgating_sections(directory: str):
    """Top 20 closest following sections for each time window and speed bucket."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise HTTPException(404, f"Directory not found: {data_dir}")

    settings_path = data_dir / "settings.json"
    nominatim_url = "http://localhost:8080"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        nominatim_url = settings.get("nominatim_url", nominatim_url)

    _DISPLAY_N = 20
    _FETCH_N = 200

    groups = []
    coords_to_geocode: set[tuple[float, float]] = set()

    speed_buckets = _discover_speed_buckets(directory)

    for speed_bucket in speed_buckets:
        segments = _collect_tailgating_segments(directory, speed_bucket)
        sections_by_window: dict[int, list[dict]] = {}
        for window_secs, _ in _TAILGATING_WINDOWS:
            sections = _find_closest_sections(segments, window_secs, top_n=_FETCH_N)
            sections_by_window[window_secs] = sections
            for s in sections:
                coords_to_geocode.add((s["mid_lat"], s["mid_lon"]))
        groups.append((speed_bucket, sections_by_window))

    geocode_cache: dict[tuple[float, float], tuple[str, str]] = {}
    async with httpx.AsyncClient() as client:
        for lat, lon in coords_to_geocode:
            geocode_cache[(lat, lon)] = await _reverse_geocode(
                client, nominatim_url, lat, lon,
            )

    result = []
    for speed_bucket, sections_by_window in groups:
        tables = []
        for window_secs, window_label in _TAILGATING_WINDOWS:
            proximity_threshold = 5 * window_secs
            kept: list[dict] = []
            for s in sections_by_window[window_secs]:
                dominated = False
                for k in kept:
                    if abs((k["mid_ts"] - s["mid_ts"]).total_seconds()) < proximity_threshold:
                        dominated = True
                        break
                if not dominated:
                    kept.append(s)

            best_by_municipality: dict[str, dict] = {}
            for s in kept:
                fallback = f"{s['mid_lat']:.5f}, {s['mid_lon']:.5f}"
                location, municipality = geocode_cache.get(
                    (s["mid_lat"], s["mid_lon"]),
                    (fallback, fallback),
                )
                row = {
                    "avg_distance_m": s["avg_distance_m"],
                    "avg_speed_kmh": s["avg_speed_kmh"],
                    "location": location,
                    "date": s["date"],
                    "start_lat": s["start_lat"],
                    "start_lon": s["start_lon"],
                    "video": s["video"],
                    "second": s["second"],
                }
                existing = best_by_municipality.get(municipality)
                if existing is None or row["avg_distance_m"] < existing["avg_distance_m"]:
                    best_by_municipality[municipality] = row
            rows = sorted(best_by_municipality.values(), key=lambda r: r["avg_distance_m"])[:_DISPLAY_N]
            tables.append({
                "window_seconds": window_secs,
                "window_label": window_label,
                "sections": rows,
            })
        result.append({
            "speed_bucket": speed_bucket,
            "speed_label": f"{speed_bucket}\u2013{speed_bucket + _BIN_SIZE} km/h",
            "tables": tables,
        })

    return {"groups": result}
