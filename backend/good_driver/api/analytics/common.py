from __future__ import annotations

import gzip
import json
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _data_dir(base: Path, filename: str) -> Path:
    return base / f"{filename}.data"


def _get_fps(ddir: Path) -> float:
    """Read FPS from metadata.json, default to 30."""
    meta_path = ddir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return float(meta.get("fps", 30))
    return 30.0


def _dist_for_second(dist_data: list, second: int, fps: float):
    """Return averaged distance entry for a given GPS second across all its frames."""
    frame_start = round(second * fps)
    frame_end = round((second + 1) * fps)
    distances = []
    for frame in range(frame_start, frame_end):
        if frame >= len(dist_data):
            break
        entry = dist_data[frame]
        if entry is not None and entry.get("distance") is not None:
            distances.append(float(entry["distance"]))
    if not distances:
        return None
    return {"distance": sum(distances) / len(distances)}


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
