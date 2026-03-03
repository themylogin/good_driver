from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .common import _collect_speed_data

router = APIRouter()


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

    # Prepend a "< min_bucket" overflow bin
    labels = [f"< {min_bucket}"] + [str(b) for b in bins[:-1]]

    # Build histogram counts: first entry is the overflow bin
    counts = [0] * len(labels)
    total = len(speeds)
    for s in speeds:
        if s <= min_bucket:
            counts[0] += 1
            continue
        for i in range(len(bins) - 1):
            if bins[i] < s <= bins[i + 1]:
                counts[i + 1] += 1
                break

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

        ax.bar(range(len(labels)), counts)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

        ax.set_title(
            f"Speed Distribution at {speed_limit} Km/h Limit (Total Travel Time: {_fmt_duration(total)})",
            fontsize=20,
        )
        ax.xaxis.set_label_text("")
        ax.yaxis.set_major_formatter(lambda x, _pos: _fmt_duration(x))

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
        for k, v in sorted(by_limit.items(), reverse=True)
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
