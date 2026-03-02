from __future__ import annotations

import gzip
import heapq
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/analytics")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _data_dir(base: Path, filename: str) -> Path:
    return base / f"{filename}.data"


def _collect_speed_data(directory: str) -> dict[int, list[float]]:
    """Read GPS speed + snap_to_road speed limit, group GPS speed_kmh by speed_limit_kmh."""
    data_dir = Path(directory)
    by_limit: dict[int, list[float]] = {}

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
        for i, entry in enumerate(snap_data):
            if entry is None:
                continue
            limit = entry.get("speed_limit_kmh")
            if limit is None:
                continue
            limit = int(limit)
            if limit < 80:
                continue
            # Use GPS speed (actual driving speed) rather than OSRM leg speed
            gps = gps_data[i] if i < len(gps_data) else None
            if gps is None:
                continue
            speed = gps.get("speed_kmh")
            if speed is None:
                continue
            by_limit.setdefault(limit, []).append(float(speed))

    return by_limit


def _render_speed_distribution_chart(speeds: list[float], speed_limit: int) -> bytes:
    """Render a cyberpunk-styled speed distribution bar chart as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplcyberpunk

    bin_size = 10
    min_bucket = max(0, (speed_limit - 40) // bin_size * bin_size)
    max_bucket = speed_limit + 80
    bins = list(range(min_bucket, max_bucket + bin_size, bin_size))
    labels = [str(b) for b in bins[:-1]]

    # Build histogram counts
    counts = [0] * len(labels)
    total = len(speeds)
    for s in speeds:
        for i in range(len(bins) - 1):
            if bins[i] < s <= bins[i + 1]:
                counts[i] += 1
                break

    with plt.style.context("cyberpunk"):
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

        ax.bar(range(len(labels)), counts)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(f"Speed limit {speed_limit} km/h: speed distribution ({total:,} samples)")
        ax.xaxis.set_label_text("")

        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            if y_value == 0:
                continue
            label = "%.2g%%" % (y_value / total * 100,)
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        mplcyberpunk.add_glow_effects(ax)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


@router.get("/speed-distribution")
async def speed_distribution(directory: str):
    """Return available speed limits with sample counts."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise HTTPException(404, f"Directory not found: {data_dir}")

    by_limit = _collect_speed_data(directory)
    limits = [
        {"speed_limit": k, "count": len(v)}
        for k, v in sorted(by_limit.items())
    ]
    return {"limits": limits}


@router.get("/speed-distribution-chart")
async def speed_distribution_chart(directory: str, speed_limit: int):
    """Return a PNG speed distribution chart for the given speed limit."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise HTTPException(404, f"Directory not found: {data_dir}")

    by_limit = _collect_speed_data(directory)
    speeds = by_limit.get(speed_limit)
    if not speeds:
        raise HTTPException(404, f"No data for speed limit {speed_limit}")

    png = _render_speed_distribution_chart(speeds, speed_limit)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Top speeding sections
# ---------------------------------------------------------------------------

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
) -> str:
    try:
        resp = await client.get(
            f"{nominatim_url.rstrip('/')}/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 16},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json().get("display_name", f"{lat:.5f}, {lon:.5f}")
    except Exception:
        pass
    return f"{lat:.5f}, {lon:.5f}"


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
    geocode_cache: dict[tuple[float, float], str] = {}
    async with httpx.AsyncClient() as client:
        for lat, lon in coords_to_geocode:
            geocode_cache[(lat, lon)] = await _reverse_geocode(
                client, nominatim_url, lat, lon,
            )

    # Build response (deduplicate by location: keep highest speed per location)
    result = []
    for speed_limit, sections_by_window in groups:
        tables = []
        for window_secs, window_label in _SPEEDING_WINDOWS:
            best_by_location: dict[str, dict] = {}
            for s in sections_by_window[window_secs]:
                location = geocode_cache.get(
                    (s["mid_lat"], s["mid_lon"]),
                    f"{s['mid_lat']:.5f}, {s['mid_lon']:.5f}",
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
                existing = best_by_location.get(location)
                if existing is None or row["avg_speed_kmh"] > existing["avg_speed_kmh"]:
                    best_by_location[location] = row
            rows = sorted(best_by_location.values(), key=lambda r: r["avg_speed_kmh"], reverse=True)[:_DISPLAY_N]
            tables.append({
                "window_seconds": window_secs,
                "window_label": window_label,
                "sections": rows,
            })
        result.append({"speed_limit": speed_limit, "tables": tables})

    return {"groups": result}
