from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...physics import safety_index
from .common import VIDEO_EXTENSIONS, _data_dir

router = APIRouter()

BIN_SIZE = 10
MIN_SPEED = 80


def _collect_safety_data(directory: str, *, no_lead_as_safe: bool = False) -> dict[int, list[float]]:
    """Collect safety index values grouped by speed bucket (10 km/h bins, >=80).

    For each frame that has both GPS speed and a following distance,
    compute the safety index and bucket it by speed.

    If *no_lead_as_safe* is True, frames with GPS speed but no lead car
    (missing distance entry) are counted as safety index 1.0.
    """
    data_dir = Path(directory)
    by_bucket: dict[int, list[float]] = {}

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
        for i in range(len(dist_data)):
            gps = gps_data[i] if i < len(gps_data) else None
            if gps is None:
                continue
            speed = gps.get("speed_kmh")
            if speed is None or speed < MIN_SPEED:
                continue

            dist_entry = dist_data[i]
            has_distance = dist_entry is not None and dist_entry.get("distance") is not None
            if has_distance:
                si = safety_index(dist_entry["distance"], speed)
            elif no_lead_as_safe:
                si = 1.0
            else:
                continue

            bucket = int(speed) // BIN_SIZE * BIN_SIZE
            by_bucket.setdefault(bucket, []).append(si)

    return by_bucket


def _render_safety_index_chart(by_bucket: dict[int, list[float]]) -> bytes:
    """Render a cyberpunk-styled safety index box plot as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplcyberpunk
    import numpy as np

    buckets = sorted(by_bucket.keys())
    labels = [f"{b}" for b in buckets]
    data = [by_bucket[b] for b in buckets]

    total_seconds = sum(len(v) for v in data)

    def _fmt_duration(seconds: int) -> str:
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes = remainder // 60
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if not parts:
            parts.append(f"{int(seconds)}s")
        return " ".join(parts)

    with plt.style.context("cyberpunk"):
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)

        medians = [float(np.median(d)) for d in data]
        means = [float(np.mean(d)) for d in data]
        p25 = [float(np.percentile(d, 25)) for d in data]
        p75 = [float(np.percentile(d, 75)) for d in data]

        x = range(len(labels))
        ax.bar(x, medians, label="Median", alpha=0.8)
        ax.plot(x, means, "o--", label="Mean", markersize=6)
        ax.plot(x, p25, "v:", label="25th percentile", markersize=5, alpha=0.7)
        ax.plot(x, p75, "^:", label="75th percentile", markersize=5, alpha=0.7)

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Driving Speed (km/h)")
        ax.set_ylabel("Safety Index")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_title(
            f"Safety Index by Driving Speed (Total: {_fmt_duration(total_seconds)})",
            fontsize=20,
        )
        ax.legend()

        for i, med in enumerate(medians):
            ax.annotate(
                f"{med:.2f}",
                (i, med),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        mplcyberpunk.add_glow_effects(ax)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.read()


@router.get("/safety-index-chart")
async def safety_index_chart(directory: str, no_lead_as_safe: bool = False):
    """Return a PNG safety index chart grouped by driving speed."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise HTTPException(404, f"Directory not found: {data_dir}")

    by_bucket = _collect_safety_data(directory, no_lead_as_safe=no_lead_as_safe)
    if not by_bucket:
        raise HTTPException(404, "No data with both GPS speed and following distance")

    png = _render_safety_index_chart(by_bucket)
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
