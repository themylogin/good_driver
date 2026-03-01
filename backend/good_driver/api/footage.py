from __future__ import annotations

import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
from skimage.morphology import skeletonize

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

    frame 0        → file_idx 0  → 0/00/00.json.gz  (covers frames 0–99)
    frame 234500   → file_idx 2345  → 0/23/45.json.gz
    frame 1234500  → file_idx 12345 → 1/23/45.json.gz
    """
    file_idx = frame_n // 100
    last2  = file_idx % 100
    mid2   = (file_idx // 100) % 100
    prefix = file_idx // 10000
    return data_dir / str(prefix) / f"{mid2:02d}" / f"{last2:02d}.json.gz"


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

_LANE_THRESHOLD = 0.5
_DRIVABLE_THRESHOLD = 0.5


_SKEL_BORDER       = 5    # discard skeleton pixels within this margin of image edges
_JUNCTION_DILATE_K = 7    # kernel size for erasing skeleton around junctions (larger = split earlier)
_RDP_EPSILON = 1.5  # Ramer-Douglas-Peucker tolerance in model-space pixels


def _trace_skeleton_branch(bxs: np.ndarray, bys: np.ndarray) -> np.ndarray:
    """Trace skeleton pixels in connectivity order, starting from an endpoint."""
    n = len(bxs)
    coords = np.column_stack([bxs, bys])
    visited = np.zeros(n, dtype=bool)

    # Build a spatial lookup: (x, y) → index
    lookup: dict[tuple[int, int], int] = {}
    for i in range(n):
        lookup[(int(bxs[i]), int(bys[i]))] = i

    # Find an endpoint (pixel with fewest skeleton neighbors) to start from.
    # Prefer a true endpoint (1 neighbor); fall back to min-y pixel.
    best_idx = 0
    best_neighbors = 9
    for i in range(n):
        x, y = int(bxs[i]), int(bys[i])
        nb = sum(
            1 for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and (x + dx, y + dy) in lookup
        )
        if nb < best_neighbors:
            best_neighbors = nb
            best_idx = i

    # Walk along the skeleton
    order = []
    cur = best_idx
    while cur is not None:
        visited[cur] = True
        order.append(cur)
        x, y = int(coords[cur, 0]), int(coords[cur, 1])
        nxt = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                j = lookup.get((x + dx, y + dy))
                if j is not None and not visited[j]:
                    nxt = j
                    break
            if nxt is not None:
                break
        cur = nxt

    return coords[order].astype(np.float32)


def _vectorize_component(
    comp_mask: np.ndarray,
    sx: float,
    sy: float,
) -> list[dict]:
    """
    Vectorize one connected component into one or more lane dicts.

    Skeletonizes the mask to get 1px-wide center lines, splits at
    junction pixels, then simplifies with RDP and scales to original
    image space.
    """
    skeleton = skeletonize(comp_mask > 0).astype(np.uint8)

    # Discard border pixels — skeleton is unreliable at image edges
    h, w = skeleton.shape
    skeleton[:_SKEL_BORDER, :] = 0
    skeleton[h - _SKEL_BORDER:, :] = 0
    skeleton[:, :_SKEL_BORDER] = 0
    skeleton[:, w - _SKEL_BORDER:] = 0

    # Find junction pixels (>2 skeleton neighbors) and remove them to split branches
    neighbor_count = cv2.filter2D(skeleton, -1, np.ones((3, 3), dtype=np.uint8)) * skeleton
    junctions = (neighbor_count > 3).astype(np.uint8)
    skel_split = skeleton.copy()
    skel_split[cv2.dilate(junctions, np.ones((_JUNCTION_DILATE_K, _JUNCTION_DILATE_K), dtype=np.uint8)) > 0] = 0

    n_branches, branch_labels = cv2.connectedComponents(skel_split, connectivity=8)

    results = []
    for branch in range(1, n_branches):
        bys, bxs = np.where(branch_labels == branch)
        if len(bys) < _MIN_UNIQUE_Y:
            continue

        # Trace skeleton pixels in connectivity order instead of sorting by Y
        polyline = _trace_skeleton_branch(bxs, bys)
        simplified = cv2.approxPolyDP(polyline, _RDP_EPSILON, closed=False)
        simplified = simplified.reshape(-1, 2)
        if len(simplified) < 2:
            continue

        # Scale to original image space
        pts = [
            [round(float(x * sx), 1), round(float(y * sy), 1)]
            for x, y in simplified
        ]
        v_vals = [p[1] for p in pts]
        results.append({
            "points": pts,
            "v_min": round(min(v_vals), 1),
            "v_max": round(max(v_vals), 1),
        })
    return results


def _mask_to_lane_lines(
    lane_logits: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> list[dict]:
    """
    Convert the lane-line segmentation output to a list of lane dicts.

    lane_logits: shape [1, C, H, W].
      - C=2: two-class logits  → argmax to get binary mask
      - C=1: single-channel probability in [0,1] → threshold at _LANE_THRESHOLD

    Returns: list of lane dicts, each with:
      - "points": [[u, v], ...]  polyline in original video pixel space
      - "cubic":  [a3, a2, a1, a0]  polynomial coefficients for u(v)
      - "v_min", "v_max": valid range for the polynomial
    """
    inner = lane_logits[0]           # [C, H, W]
    if inner.shape[0] == 1:
        mask = (inner[0] > _LANE_THRESHOLD).astype(np.uint8)
    else:
        mask = inner.argmax(axis=0).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    sx = orig_w / _INPUT_W
    sy = orig_h / _INPUT_H
    lanes: list[dict] = []

    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < _MIN_LANE_PIXELS:
            continue
        comp_mask = (labels == lbl).astype(np.uint8)
        lanes.extend(_vectorize_component(comp_mask, sx, sy))

    return lanes


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
            lane_dicts = _mask_to_lane_lines(outputs[1], orig_w, orig_h)
            lane_lines = [ld["points"] for ld in lane_dicts]
        except Exception as e:
            logger.warning("Lane vectorization failed: %s", e)

    return {
        "detections": detections,
        "lane_lines": lane_lines,
    }


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
                fpath.write_bytes(gzip.compress(json.dumps(batch).encode()))
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
    return json.loads(gzip.decompress(fpath.read_bytes()))


@router.get("/debug-frame")
async def debug_frame(filename: str, directory: str, frame: int):
    """Return a JPEG with raw segmentation masks + lane polylines overlaid on the frame."""
    path = Path(directory) / filename
    if not path.exists():
        raise HTTPException(404, f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(500, f"Cannot open video: {path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, frame_bgr = cap.read()
        if not ok:
            raise HTTPException(400, f"Cannot read frame {frame}")
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    # Run model
    session = _get_session()
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor, sx, sy = _preprocess(img)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})

    # Build overlay on the original frame (RGB)
    base = np.array(img, dtype=np.float32)
    overlay = base.copy()
    alpha = 0.4

    # Driveable area mask → green
    if len(outputs) >= 5 and outputs[0] is not None:
        seg_inner = outputs[0][0]
        if seg_inner.shape[0] == 1:
            seg_mask = (seg_inner[0] > _DRIVABLE_THRESHOLD).astype(np.uint8)
        else:
            seg_mask = seg_inner.argmax(axis=0).astype(np.uint8)
        seg_full = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        driveable = seg_full == 1
        if driveable.any():
            overlay[driveable] = base[driveable] * (1 - alpha) + np.array([0, 255, 0]) * alpha

    # Lane mask → red overlay + yellow skeleton pixels
    ll_mask = None
    if len(outputs) >= 5 and outputs[1] is not None:
        ll_inner = outputs[1][0]
        if ll_inner.shape[0] == 1:
            ll_mask = (ll_inner[0] > _LANE_THRESHOLD).astype(np.uint8)
        else:
            ll_mask = ll_inner.argmax(axis=0).astype(np.uint8)
        ll_full = cv2.resize(ll_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        lanes = ll_full == 1
        if lanes.any():
            overlay[lanes] = base[lanes] * (1 - alpha) + np.array([255, 0, 0]) * alpha

    result = overlay.astype(np.uint8)

    # Draw raw skeleton pixels (yellow) — reuse ll_mask from above
    if ll_mask is not None:
        try:
            n_lbl, lbl_map, st, _ = cv2.connectedComponentsWithStats(ll_mask, connectivity=8)
            sx = orig_w / _INPUT_W
            sy = orig_h / _INPUT_H
            for lbl in range(1, n_lbl):
                if st[lbl, cv2.CC_STAT_AREA] < _MIN_LANE_PIXELS:
                    continue
                comp = (lbl_map == lbl).astype(np.uint8)
                skel = skeletonize(comp > 0).astype(np.uint8)
                skel_full = cv2.resize(skel, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                result[skel_full == 1] = (255, 255, 0)
        except Exception:
            pass

    # Encode as JPEG
    rgb_to_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    _, jpeg = cv2.imencode(".jpg", rgb_to_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@router.get("/process-frame")
async def process_frame(filename: str, directory: str, frame: int):
    """Run inference on a single video frame without writing results to disk."""
    path = Path(directory) / filename
    if not path.exists():
        raise HTTPException(404, f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(500, f"Cannot open video: {path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, frame_bgr = cap.read()
        if not ok:
            raise HTTPException(400, f"Cannot read frame {frame}")
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    session = _get_session()
    result = _infer_frame(frame_bgr, orig_w, orig_h, session)
    # Assign sequential track_ids so the frontend gives each car a different color
    for i, det in enumerate(result["detections"]):
        det["track_id"] = i
    result["frame"] = frame
    return result


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
