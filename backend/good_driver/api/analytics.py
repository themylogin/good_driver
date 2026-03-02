from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

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
