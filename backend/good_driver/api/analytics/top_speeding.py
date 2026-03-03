from __future__ import annotations

import gzip
import heapq
import json
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException

from .common import VIDEO_EXTENSIONS, _data_dir

router = APIRouter()

_SPEEDING_SPEED_LIMITS = [120, 100]
_SPEEDING_WINDOWS = [
    (60, "1 minute"),
    (300, "5 minutes"),
    (600, "10 minutes"),
    (1800, "30 minutes"),
    (3600, "1 hour"),
]


def _collect_speed_segments(
    directory: str, speed_limit: int,
) -> list[list[dict]]:
    """Collect contiguous runs of seconds at *speed_limit* across all videos.

    All videos are merged into one timeline sorted by timestamp.  A new
    segment starts when the speed limit doesn't match, data is missing,
    or there is a timestamp gap > 5 s.
    """
    data_dir = Path(directory)

    # 1. Gather all data points with timestamps from every video
    all_points: list[dict] = []
    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        ddir = _data_dir(data_dir, f.name)
        snap_path = ddir / "snap_to_road.json.gz"
        gps_path = ddir / "gps.json.gz"
        if not snap_path.exists() or not gps_path.exists():
            continue
        snap_data = json.loads(gzip.decompress(snap_path.read_bytes()))
        gps_data = json.loads(gzip.decompress(gps_path.read_bytes()))
        for i, snap_entry in enumerate(snap_data):
            if snap_entry is None:
                continue
            ts_str = snap_entry.get("timestamp")
            if ts_str is None:
                continue
            gps = gps_data[i] if i < len(gps_data) else None
            if gps is None or gps.get("speed_kmh") is None:
                continue
            all_points.append({
                "ts": datetime.fromisoformat(ts_str.replace("Z", "+00:00")),
                "speed_kmh": float(gps["speed_kmh"]),
                "speed_limit_kmh": snap_entry.get("speed_limit_kmh"),
                "lat": snap_entry["lat"],
                "lon": snap_entry["lon"],
                "video": f.name,
                "second": i,
            })

    all_points.sort(key=lambda p: p["ts"])

    # 2. Split into contiguous segments by speed limit match + timestamp gap.
    #    The last second of each OSRM-matched video has speed_limit=None
    #    (no "next" leg), so when limit is None and the gap is small we
    #    carry forward the previous limit to avoid breaking segments.
    segments: list[list[dict]] = []
    current: list[dict] = []
    prev_ts: datetime | None = None
    prev_limit: int | None = None

    for pt in all_points:
        raw_limit = pt["speed_limit_kmh"]
        limit = int(raw_limit) if raw_limit is not None else None

        # Carry forward previous limit across small gaps when limit is None
        gap_secs = (pt["ts"] - prev_ts).total_seconds() if prev_ts is not None else 0
        if limit is None and gap_secs <= 5 and prev_limit is not None:
            limit = prev_limit

        matches = limit is not None and limit == speed_limit
        gap = gap_secs > 5

        if not matches or gap:
            if current:
                segments.append(current)
                current = []

        if matches:
            current.append({
                "speed_kmh": pt["speed_kmh"],
                "lat": pt["lat"],
                "lon": pt["lon"],
                "ts": pt["ts"],
                "video": pt["video"],
                "second": pt["second"],
            })

        prev_ts = pt["ts"]
        if limit is not None:
            prev_limit = limit

    if current:
        segments.append(current)

    return segments


def _find_top_sections(
    segments: list[list[dict]],
    window_seconds: int,
    top_n: int = 20,
) -> list[dict]:
    """Sliding-window search for the top-N fastest sections."""
    # Min-heap of (avg_speed, counter, info_dict)
    heap: list[tuple[float, int, dict]] = []
    counter = 0

    for entries in segments:
        n = len(entries)
        if n < window_seconds:
            continue

        running_sum = sum(e["speed_kmh"] for e in entries[:window_seconds])

        for start in range(n - window_seconds + 1):
            if start > 0:
                running_sum += entries[start + window_seconds - 1]["speed_kmh"]
                running_sum -= entries[start - 1]["speed_kmh"]

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
            }

            if len(heap) < top_n:
                heapq.heappush(heap, (avg, counter, info))
                counter += 1
            elif avg > heap[0][0]:
                heapq.heapreplace(heap, (avg, counter, info))
                counter += 1

    results = sorted(heap, key=lambda t: t[0], reverse=True)
    return [
        {
            "avg_speed_kmh": round(avg, 1),
            "mid_lat": info["mid_lat"],
            "mid_lon": info["mid_lon"],
            "mid_ts": datetime.fromisoformat(info["mid_ts"]),
            "date": datetime.fromisoformat(info["mid_ts"]).astimezone().strftime("%Y-%m-%d %H:%M"),
            "start_lat": info["start_lat"],
            "start_lon": info["start_lon"],
            "video": info["video"],
            "second": info["second"],
        }
        for avg, _, info in results
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


@router.get("/top-speeding-sections")
async def top_speeding_sections(directory: str):
    """Top 20 fastest road sections for each time window and speed limit."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise HTTPException(404, f"Directory not found: {data_dir}")

    # Read Nominatim URL from settings
    settings_path = data_dir / "settings.json"
    nominatim_url = "http://localhost:8080"
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        nominatim_url = settings.get("nominatim_url", nominatim_url)

    # Compute top sections for each speed limit and window.
    # Fetch extra candidates so that after deduplication by location we still
    # have up to 20 rows.
    _DISPLAY_N = 20
    _FETCH_N = 200

    groups = []
    coords_to_geocode: set[tuple[float, float]] = set()

    for speed_limit in _SPEEDING_SPEED_LIMITS:
        segments = _collect_speed_segments(directory, speed_limit)
        sections_by_window: dict[int, list[dict]] = {}
        for window_secs, _ in _SPEEDING_WINDOWS:
            sections = _find_top_sections(segments, window_secs, top_n=_FETCH_N)
            sections_by_window[window_secs] = sections
            for s in sections:
                coords_to_geocode.add((s["mid_lat"], s["mid_lon"]))
        groups.append((speed_limit, sections_by_window))

    # Reverse-geocode all unique coordinates
    geocode_cache: dict[tuple[float, float], tuple[str, str]] = {}
    async with httpx.AsyncClient() as client:
        for lat, lon in coords_to_geocode:
            geocode_cache[(lat, lon)] = await _reverse_geocode(
                client, nominatim_url, lat, lon,
            )

    # Build response
    result = []
    for speed_limit, sections_by_window in groups:
        tables = []
        for window_secs, window_label in _SPEEDING_WINDOWS:
            # 1) Deduplicate by proximity: sections whose timestamps
            #    are within 5x the window are overlapping;
            #    keep only the fastest.
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

            # 2) Deduplicate by municipality: keep highest speed per municipality.
            best_by_municipality: dict[str, dict] = {}
            for s in kept:
                fallback = f"{s['mid_lat']:.5f}, {s['mid_lon']:.5f}"
                location, municipality = geocode_cache.get(
                    (s["mid_lat"], s["mid_lon"]),
                    (fallback, fallback),
                )
                row = {
                    "avg_speed_kmh": s["avg_speed_kmh"],
                    "location": location,
                    "date": s["date"],
                    "start_lat": s["start_lat"],
                    "start_lon": s["start_lon"],
                    "video": s["video"],
                    "second": s["second"],
                }
                existing = best_by_municipality.get(municipality)
                if existing is None or row["avg_speed_kmh"] > existing["avg_speed_kmh"]:
                    best_by_municipality[municipality] = row
            rows = sorted(best_by_municipality.values(), key=lambda r: r["avg_speed_kmh"], reverse=True)[:_DISPLAY_N]
            tables.append({
                "window_seconds": window_secs,
                "window_label": window_label,
                "sections": rows,
            })
        result.append({"speed_limit": speed_limit, "tables": tables})

    return {"groups": result}
