from __future__ import annotations

import gzip
import json
from pathlib import Path

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
