from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from scipy.interpolate import UnivariateSpline

from .calibrate import (
    _CONF_THRESHOLD,
    _INPUT_H,
    _INPUT_W,
    _decode_postnms_detections,
    _decode_raw_predictions,
    _get_session,
    _nms,
    _preprocess,
    _scale_boxes,
)
from ..tracker import ByteTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/footage")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# ---------------------------------------------------------------------------
# Thread pool — max_workers=1 keeps processing sequential
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=1)
_processing_videos: set[str] = set()   # "directory|filename" keys currently queued or running
_live_progress: dict[str, int] = {}    # key → current frame_n (updated every frame)


# ---------------------------------------------------------------------------
# File-path helpers
# ---------------------------------------------------------------------------

def _frame_file_path(data_dir: Path, frame_n: int) -> Path:
    """
    Encode frame_n into a 3-level tiled path.

    frame 0        → file_idx 0  → 0/00/00.json  (covers frames 0–99)
    frame 234500   → file_idx 2345  → 0/23/45.json
    frame 1234500  → file_idx 12345 → 1/23/45.json
    """
    file_idx = frame_n // 100
    last2  = file_idx % 100
    mid2   = (file_idx // 100) % 100
    prefix = file_idx // 10000
    return data_dir / str(prefix) / f"{mid2:02d}" / f"{last2:02d}.json"


def _data_dir(base: Path, filename: str) -> Path:
    return base / f"{filename}.data"


def _read_metadata(data_dir: Path) -> dict | None:
    p = data_dir / "metadata.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _write_metadata(data_dir: Path, total: int, processed: int, **extra) -> None:
    """Write metadata, preserving any existing extra fields unless overridden by **extra."""
    meta_path = data_dir / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta.update({"total_frames": total, "processed_frames": processed})
    meta.update(extra)
    meta_path.write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# Lane-line vectorization
# ---------------------------------------------------------------------------

_MIN_LANE_PIXELS = 30   # ignore tiny blobs
_MIN_UNIQUE_Y    = 5    # need vertical spread for a meaningful spline
_LANE_SAMPLES    = 20   # number of points per vectorised lane


_LANE_THRESHOLD = 0.5


def _mask_to_lane_lines(
    lane_logits: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> list[list[list[float]]]:
    """
    Convert the lane-line segmentation output to a list of smooth polylines.

    lane_logits: shape [1, C, H, W].
      - C=2: two-class logits  → argmax to get binary mask
      - C=1: single-channel probability in [0,1] → threshold at _LANE_THRESHOLD
    Returns: [[[x, y], ...], ...]  in original video pixel space.
    """
    inner = lane_logits[0]           # [C, H, W]
    if inner.shape[0] == 1:
        # Single-channel probability map (post-sigmoid)
        mask = (inner[0] > _LANE_THRESHOLD).astype(np.uint8)
    else:
        mask = inner.argmax(axis=0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    sx = orig_w / _INPUT_W
    sy = orig_h / _INPUT_H
    lines: list[list[list[float]]] = []

    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < _MIN_LANE_PIXELS:
            continue

        ys, xs = np.where(labels == lbl)

        if len(np.unique(ys)) < _MIN_UNIQUE_Y:
            continue

        # Average x per unique y row (avoids duplicate-y issues for the spline)
        unique_ys = np.unique(ys)
        mean_xs = np.array([xs[ys == y].mean() for y in unique_ys], dtype=float)

        if len(unique_ys) < _MIN_UNIQUE_Y:
            continue

        try:
            k = min(2, len(unique_ys) - 1)
            spline = UnivariateSpline(unique_ys, mean_xs, k=k, s=len(unique_ys) * 4)
        except Exception:
            continue

        y_samples = np.linspace(float(unique_ys[0]), float(unique_ys[-1]), _LANE_SAMPLES)
        x_samples = spline(y_samples)

        pts = [
            [round(float(x * sx), 1), round(float(y * sy), 1)]
            for x, y in zip(x_samples, y_samples)
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(pts) < 2:
            continue
        lines.append(pts)

    return lines


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def _infer_frame(
    frame_bgr: np.ndarray,
    orig_w: int,
    orig_h: int,
    session,
) -> dict:
    """Run YOLOPv2 on a single BGR frame. Returns detections + lane_lines."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor, sx, sy = _preprocess(img)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})

    # ── Detections ──
    det_output = None
    for o in outputs:
        if isinstance(o, np.ndarray) and o.ndim == 3 and o.shape[-1] in (6, 7):
            det_output = o
            break

    if det_output is not None:
        raw_boxes = _decode_postnms_detections(det_output)
        boxes = _nms(raw_boxes)
    elif len(outputs) >= 5:
        raw_boxes = _decode_raw_predictions(outputs)
        boxes = _nms(raw_boxes)
    else:
        boxes = []

    scaled = _scale_boxes(boxes, sx, sy)
    detections = []
    for b in scaled:
        x1 = max(0, int(b["x1"]))
        y1 = max(0, int(b["y1"]))
        x2 = min(orig_w, int(b["x2"]))
        y2 = min(orig_h, int(b["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue
        if b["confidence"] < _CONF_THRESHOLD:
            continue
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": round(b["confidence"], 3),
        })

    # ── Lane lines ──
    # outputs[1] is the lane-line segmentation head when len(outputs) >= 5
    lane_lines: list = []
    if len(outputs) >= 5 and outputs[1] is not None:
        try:
            lane_lines = _mask_to_lane_lines(outputs[1], orig_w, orig_h)
        except Exception as e:
            logger.warning("Lane vectorization failed: %s", e)

    return {"detections": detections, "lane_lines": lane_lines}


# ---------------------------------------------------------------------------
# Background processing worker
# ---------------------------------------------------------------------------

def _process_video_worker(filename: str, directory: str, key: str) -> None:
    """Process all frames of a video and write frame-data JSON files."""
    try:
        data_dir = Path(directory)
        path = data_dir / filename
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", path)
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ddir = _data_dir(data_dir, filename)
        ddir.mkdir(exist_ok=True)

        # Resume from last completed batch boundary
        meta = _read_metadata(ddir)
        resume_frame = 0
        if meta is not None:
            # Round down to nearest batch of 100 to resume cleanly
            resume_frame = (meta["processed_frames"] // 100) * 100
        _write_metadata(ddir, total, resume_frame,
                        fps=cap.get(cv2.CAP_PROP_FPS),
                        width=orig_w, height=orig_h)

        # Seek to resume position
        if resume_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
            logger.info("Resuming %s from frame %d/%d", filename, resume_frame, total)

        session = _get_session()
        tracker = ByteTracker()

        batch: list[dict] = []
        frame_n = resume_frame

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = _infer_frame(frame, orig_w, orig_h, session)
            result["detections"] = tracker.update(result["detections"])
            result["frame"] = frame_n
            batch.append(result)

            if len(batch) == 100:
                first_frame = frame_n - 99
                fpath = _frame_file_path(ddir, first_frame)
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(json.dumps(batch))
                _write_metadata(ddir, total, frame_n + 1)
                batch = []

            _live_progress[key] = frame_n + 1
            frame_n += 1

        # Flush remainder (< 100 frames)
        if batch:
            first_frame = frame_n - len(batch)
            fpath = _frame_file_path(ddir, first_frame)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(json.dumps(batch))

        _write_metadata(ddir, total, frame_n)
        cap.release()
        logger.info("Finished processing %s (%d frames)", filename, frame_n)

    except Exception:
        logger.exception("Error processing video %s", filename)
    finally:
        _live_progress.pop(key, None)
        _processing_videos.discard(key)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _video_entry(directory: Path, filename: str) -> dict:
    """Build a video entry dict with a proxy URL served by the backend."""
    from urllib.parse import quote
    video_url = f"/api/footage/video?filename={quote(filename, safe='')}&directory={quote(str(directory), safe='')}"
    return {"filename": filename, "video_url": video_url}


@router.get("/list")
async def list_videos(directory: str):
    """Return all video files in the given directory."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise HTTPException(404, f"Directory not found: {data_dir}")

    videos = sorted(
        f.name
        for f in data_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    return {"videos": [_video_entry(data_dir, v) for v in videos]}


@router.get("/video")
async def serve_video(filename: str, directory: str):
    """Serve a video file with range-request support for seeking."""
    data_dir = Path(directory)
    path = (data_dir / filename).resolve()
    if not str(path).startswith(str(data_dir.resolve())):
        raise HTTPException(403, "Access denied")
    if not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


@router.get("/metadata")
async def get_metadata(filename: str, directory: str):
    """Return processing metadata for a video."""
    ddir = _data_dir(Path(directory), filename)
    meta = _read_metadata(ddir)
    if meta is None:
        return {"total_frames": 0, "processed_frames": 0}
    # Overlay live frame counter for more granular progress
    key = f"{directory}|{filename}"
    live = _live_progress.get(key)
    if live is not None and live > meta.get("processed_frames", 0):
        meta = {**meta, "processed_frames": live}
    meta["processing"] = key in _processing_videos
    return meta


@router.get("/frames")
async def get_frames(filename: str, batch: int, directory: str):
    """Return precomputed frame data for a batch of 100 frames.

    batch: integer batch index (frame_n // 100), e.g. batch=5 covers frames 500-599.
    Returns a JSON array of up to 100 frame dicts, each with 'frame', 'detections', 'lane_lines'.
    """
    ddir = _data_dir(Path(directory), filename)
    fpath = _frame_file_path(ddir, batch * 100)
    if not fpath.exists():
        raise HTTPException(404, f"Batch {batch} not yet processed for {filename!r}")
    return json.loads(fpath.read_text())


@router.post("/start-processing")
async def start_processing(directory: str):
    """Submit all unprocessed videos to the background worker pool."""
    data_dir = Path(directory)
    if not data_dir.exists():
        raise HTTPException(404, f"Directory not found: {data_dir}")

    started: list[str] = []
    videos = sorted(
        f.name
        for f in data_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    )

    for filename in videos:
        key = f"{directory}|{filename}"
        if key in _processing_videos:
            continue  # already queued or running

        ddir = _data_dir(data_dir, filename)
        meta = _read_metadata(ddir)
        if meta is not None and meta["processed_frames"] >= meta["total_frames"] > 0:
            continue  # already complete

        _processing_videos.add(key)
        _executor.submit(_process_video_worker, filename, directory, key)
        started.append(filename)

    return {"started": started}
