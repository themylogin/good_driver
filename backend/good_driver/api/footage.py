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
import math

from .calibrate import (
    _CONF_THRESHOLD,
    _INPUT_H,
    _INPUT_W,
    _collect_measurements,
    _decode_postnms_detections,
    _decode_raw_predictions,
    _get_session,
    _nms,
    _preprocess,
    _scale_boxes,
    solve_camera_params,
)
from ..segmentation_mask import encode_mask
from ..tracker import ByteTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/footage")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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

_LANE_THRESHOLD = 0.5
_DRIVABLE_THRESHOLD = 0.5

_BEV_BOTTOM_CROP   = 10   # model-space rows to ignore at the bottom of driveable area
_EGO_TOP_CROP      = 5    # model-space rows to ignore at the top of ego lane
_LATERAL_VEL_OUTLIER_MULT = 5  # centerline |dX/dZ| spike threshold as multiple of median
_LATERAL_VEL_MIN_THRESH   = 0.3  # absolute floor for the spike threshold (handles straight lanes)



def _image_to_world(
    u: float, v: float,
    video_w: int, video_h: int,
    cam: dict,
) -> tuple[float, float] | None:
    """Back-project image pixel (u, v) to world ground-plane (X, Z)."""
    fx_scaled = cam["fx"] * (video_w / cam["image_width"])
    cx = video_w / 2
    cy = video_h / 2
    pitch_rad = cam["pitch_degrees"] * math.pi / 180
    beta = math.atan((v - cy) / fx_scaled)
    angle = beta + pitch_rad
    if math.tan(angle) <= 0:
        return None
    Z = cam["camera_height_m"] / math.tan(angle)
    X = (u - cx) * Z / fx_scaled
    return (X, Z)


def _world_to_image(
    X: float, Z: float,
    video_w: int, video_h: int,
    cam: dict,
) -> tuple[float, float] | None:
    """Project world ground-plane (X, Z) back to image pixel (u, v)."""
    if Z <= 0:
        return None
    fx_scaled = cam["fx"] * (video_w / cam["image_width"])
    cx = video_w / 2
    cy = video_h / 2
    pitch_rad = cam["pitch_degrees"] * math.pi / 180
    angle = math.atan(cam["camera_height_m"] / Z)
    beta = angle - pitch_rad
    v = cy + fx_scaled * math.tan(beta)
    u = cx + X * fx_scaled / Z
    return (u, v)


def _load_camera_params(directory: Path) -> dict | None:
    """Try to load solved camera params for the given directory."""
    try:
        measurements = _collect_measurements(directory)
        if len(measurements) < 2:
            return None
        iw = measurements[0]["image_width"]
        ih = measurements[0]["image_height"]
        result = solve_camera_params(measurements, iw, ih)
        result["image_width"] = iw
        result["image_height"] = ih
        return result
    except Exception:
        return None



def _extract_masks(outputs) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode model outputs into driveable-area and lane-separator masks.

    Returns (da_mask, lane_mask) at model resolution, or None if unavailable.
    Both are uint8 arrays with values 0 or 1.
    """
    if len(outputs) < 5 or outputs[0] is None:
        return None

    seg_inner = outputs[0][0]
    if seg_inner.shape[0] == 1:
        seg_mask = (seg_inner[0] > _DRIVABLE_THRESHOLD).astype(np.uint8)
    else:
        seg_mask = seg_inner.argmax(axis=0).astype(np.uint8)
    da_mask = (seg_mask == 1).astype(np.uint8)

    lane_mask = np.zeros_like(seg_mask)
    if outputs[1] is not None:
        ll_inner = outputs[1][0]
        if ll_inner.shape[0] == 1:
            lane_mask = (ll_inner[0] > _LANE_THRESHOLD).astype(np.uint8)
        else:
            lane_mask = (ll_inner.argmax(axis=0) > 0).astype(np.uint8)
        lane_mask = cv2.dilate(lane_mask, np.ones((3, 3), np.uint8), iterations=1)

    return da_mask, lane_mask


def _build_lane_polygons(
    outputs, orig_w: int, orig_h: int,
) -> tuple[list[np.ndarray], int | None, np.ndarray | None]:
    """Split driveable area by lane separators into individual lane masks.

    Returns (masks, ego_index, ego_discarded) where masks is a list of bool
    arrays at original resolution, ego_index is the index of the ego lane
    (or None), and ego_discarded is a bool mask of discarded ego-lane pixels
    (bottom crop + edge-touching rows) at original resolution.
    """
    masks_result = _extract_masks(outputs)
    if masks_result is None:
        return [], None, None

    da_model, lane_cut = masks_result

    da_cut = (da_model & (1 - lane_cut)).astype(np.uint8)
    da_cut[_INPUT_H - _BEV_BOTTOM_CROP:, :] = 0
    n_labels, da_labels_model = cv2.connectedComponents(da_cut, connectivity=4)

    mc = (_INPUT_H - _BEV_BOTTOM_CROP - 1, _INPUT_W // 2)
    ego_label = da_labels_model[mc[0], mc[1]]

    masks = []
    ego_index = None
    ego_discarded: np.ndarray | None = None
    for lbl in range(1, n_labels):
        comp = (da_labels_model == lbl).astype(np.uint8)
        # Dilate to absorb lane-cut gap, clip to driveable area excluding lane separators
        comp = cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=1) & da_cut
        comp[_INPUT_H - _BEV_BOTTOM_CROP:, :] = 0

        if lbl == ego_label:
            # Build the full ego mask (including discarded) at model resolution
            ego_full_model = comp.copy()
            # Also include the bottom-cropped area from da_model for this label
            da_cut_no_bottom = (da_model & (1 - lane_cut)).astype(np.uint8)
            ego_bottom = da_cut_no_bottom.copy()
            ego_bottom[:_INPUT_H - _BEV_BOTTOM_CROP, :] = 0
            ego_full_model = ego_full_model | ego_bottom

            # Discard rows where ego comp touches edge (col <= 1 or col >= W-2)
            edge_rows = set()
            for r in range(_INPUT_H):
                if comp[r].any():
                    xs = np.where(comp[r])[0]
                    if xs[0] <= 1 or xs[-1] >= _INPUT_W - 2:
                        edge_rows.add(r)

            # Only apply edge discard if <80% of rows would be removed
            all_active = np.where(comp.any(axis=1))[0]
            if len(all_active) > 0 and len(edge_rows) / len(all_active) < 0.8:
                for r in edge_rows:
                    comp[r, :] = 0

            # Discard narrow rows (<4 px) from the top of the ego lane
            active_rows = np.where(comp.any(axis=1))[0]
            for r in active_rows:
                if comp[r].sum() < 4:
                    comp[r, :] = 0
                else:
                    break

            # Crop top N rows of ego lane
            active_rows = np.where(comp.any(axis=1))[0]
            if len(active_rows) > _EGO_TOP_CROP:
                for r in active_rows[:_EGO_TOP_CROP]:
                    comp[r, :] = 0

            # Discarded = everything in full ego model mask that isn't in the valid comp
            disc_model = (ego_full_model.astype(bool) & ~comp.astype(bool)).astype(np.uint8)
            ego_discarded = cv2.resize(disc_model, (orig_w, orig_h),
                                       interpolation=cv2.INTER_NEAREST) > 0

        full = cv2.resize(comp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0
        if not full.any() and lbl != ego_label:
            continue
        if lbl == ego_label:
            ego_index = len(masks)
        masks.append(full)

    return masks, ego_index, ego_discarded


def _project_lane_boundaries(
    lane_masks: list[np.ndarray], orig_w: int, orig_h: int, cam: dict,
) -> tuple[list[tuple[list[tuple[float, float]], list[tuple[float, float]]]], list[tuple[float, float]]]:
    """Project lane polygon boundaries to world space.

    Returns (lane_polys_world, all_world_pts) where each entry in
    lane_polys_world is (left_pts, right_pts) as lists of (X, Z) pairs.
    """
    lane_polys_world: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]] = []
    all_world: list[tuple[float, float]] = []
    for mask in lane_masks:
        left_pts: list[tuple[float, float]] = []
        right_pts: list[tuple[float, float]] = []
        rows = np.where(mask.any(axis=1))[0]
        for r in rows[::4]:
            xs = np.where(mask[r, :])[0]
            if len(xs) == 0:
                continue
            wl = _image_to_world(int(xs[0]), int(r), orig_w, orig_h, cam)
            wr = _image_to_world(int(xs[-1]), int(r), orig_w, orig_h, cam)
            if wl and wr and wl[1] > 0 and wr[1] > 0:
                left_pts.append((wl[0], wl[1]))
                right_pts.append((wr[0], wr[1]))
                all_world.append((wl[0], wl[1]))
                all_world.append((wr[0], wr[1]))
        lane_polys_world.append((left_pts, right_pts))
    return lane_polys_world, all_world


def _build_lane_interps(
    lane_polys_world: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None]:
    """Build sorted interpolation arrays for each lane polygon."""
    interps: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = []
    for left_pts, right_pts in lane_polys_world:
        if len(left_pts) < 2:
            interps.append(None)
            continue
        zl = np.array([p[1] for p in left_pts])
        xl = np.array([p[0] for p in left_pts])
        zr = np.array([p[1] for p in right_pts])
        xr = np.array([p[0] for p in right_pts])
        order_l = np.argsort(zl)
        order_r = np.argsort(zr)
        interps.append((zl[order_l], xl[order_l], zr[order_r], xr[order_r]))
    return interps


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def _extract_detections(
    outputs: list, sx: float, sy: float, orig_w: int, orig_h: int,
) -> list[dict]:
    """Extract car detections from existing model outputs (no re-run)."""
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
    return detections


def _infer_frame(
    frame_bgr: np.ndarray,
    orig_w: int,
    orig_h: int,
    session,
) -> dict:
    """Run YOLOPv2 on a single BGR frame. Returns detections + segmentation mask."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor, sx, sy = _preprocess(img)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})

    detections = _extract_detections(outputs, sx, sy, orig_w, orig_h)

    mask_payload = None
    masks_result = _extract_masks(outputs)
    if masks_result is not None:
        da_mask, lane_mask = masks_result
        mask_payload = encode_mask(da_mask, lane_mask, width=_INPUT_W)

    result: dict = {
        "detections": detections,
    }
    if mask_payload is not None:
        result["mask"] = mask_payload

    return result


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
    Returns a JSON array of up to 100 frame dicts, each with 'frame', 'detections', 'mask'.
    """
    ddir = _data_dir(Path(directory), filename)
    fpath = _frame_file_path(ddir, batch * 100)
    if not fpath.exists():
        raise HTTPException(404, f"Batch {batch} not yet processed for {filename!r}")
    frames = json.loads(gzip.decompress(fpath.read_bytes()))
    return frames


def _eval_hinge(z: float, coeffs, mode: str, zb: float) -> float:
    """Evaluate the centerline hinge model at a given Z."""
    m, n, a = coeffs
    if mode == "straight":
        return m * z + n
    elif mode == "curve":
        return m * z + n + a * z * z
    elif mode == "straight_then_curve":
        return m * z + n + a * max(0, z - zb) ** 2
    else:  # curve_then_straight
        return m * z + n + a * max(0, zb - z) ** 2


def _render_debug(frame_bgr: np.ndarray, cam_directory: Path) -> bytes:
    """Render debug overlay on a BGR frame and return JPEG bytes."""
    orig_h, orig_w = frame_bgr.shape[:2]

    session = _get_session()
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor, sx, sy = _preprocess(img)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})

    base = np.array(img, dtype=np.float32)
    overlay = base.copy()
    alpha = 0.4

    lane_masks, ego_idx, ego_discarded = _build_lane_polygons(outputs, orig_w, orig_h)

    if ego_idx is not None:
        ego_full = lane_masks[ego_idx]
        overlay[ego_full] = base[ego_full] * (1 - alpha) + np.array([0, 255, 0]) * alpha
        if ego_discarded is not None and ego_discarded.any():
            alpha_dim = 0.12
            overlay[ego_discarded] = base[ego_discarded] * (1 - alpha_dim) + np.array([0, 255, 0]) * alpha_dim
        for i, lm in enumerate(lane_masks):
            if i != ego_idx and lm.any():
                overlay[lm] = base[lm] * (1 - alpha) + np.array([0, 100, 255]) * alpha

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

    detections = _extract_detections(outputs, sx, sy, orig_w, orig_h)
    cam = _load_camera_params(cam_directory)
    lane_polys_world: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]] = []
    if cam is not None and lane_masks:
        lane_polys_world, _ = _project_lane_boundaries(lane_masks, orig_w, orig_h, cam)
    lane_interps = _build_lane_interps(lane_polys_world)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # Build ego lane centerline (world-space hinge model)
    centerline_img: list[tuple[int, int]] = []
    raw_centerline_img: list[tuple[int, int]] = []
    best_coeffs = None
    best_zb = 0.0
    best_mode = "straight_then_curve"
    if ego_idx is not None and ego_idx < len(lane_interps) and cam is not None:
        interp = lane_interps[ego_idx]
        if interp is not None:
            zl, xl, zr, xr = interp
            # Raw centerline: midpoint of left/right boundaries at each sample Z
            z_min = max(zl[0], zr[0])
            z_max = min(zl[-1], zr[-1])
            z_samples = np.linspace(z_min, z_max, 100)
            cx_raw = np.array([
                (np.interp(z, zl, xl) + np.interp(z, zr, xr)) / 2.0
                for z in z_samples
            ])
            # Outlier rejection: truncate at first lateral velocity spike
            _dx = np.diff(cx_raw)
            _dz = np.diff(z_samples)
            _lat_vel = np.abs(_dx / _dz)
            _med_vel = max(float(np.median(_lat_vel)), 0.01)
            _thresh = max(_LATERAL_VEL_OUTLIER_MULT * _med_vel, _LATERAL_VEL_MIN_THRESH)
            _cutoff = len(cx_raw)
            for _i in range(len(_lat_vel)):
                if _lat_vel[_i] > _thresh:
                    _cutoff = _i
                    break
            _start = 0
            for _i in range(min(_cutoff, len(_lat_vel)) - 1, -1, -1):
                if _lat_vel[_i] > _thresh:
                    _start = _i + 1
                    break
            cx_trimmed = cx_raw[_start:_cutoff]
            z_trimmed = z_samples[_start:_cutoff]
            # Orange line: raw (untrimmed) points
            for i, z in enumerate(z_samples):
                uv = _world_to_image(float(cx_raw[i]), float(z), orig_w, orig_h, cam)
                if uv and 0 <= uv[0] < orig_w and 0 <= uv[1] < orig_h:
                    raw_centerline_img.append((int(round(uv[0])), int(round(uv[1]))))
            # Trimmed centers for hinge fit
            world_centers: list[tuple[float, float]] = []
            for i, z in enumerate(z_trimmed):
                world_centers.append((float(cx_trimmed[i]), float(z)))
            # Fitted centerline: linear+parabolic hinge model
            # X(Z) = m*Z + n + a*max(0, Z-Zb)^2  (straight then curves)
            # or   = m*Z + n + a*max(0, Zb-Z)^2  (curves then straightens)
            # For fixed Zb, linear in (m, n, a) — search over Zb.
            if len(world_centers) >= 3:
                zs = np.array([p[1] for p in world_centers])
                x_vals = np.array([p[0] for p in world_centers])

                best_err = float("inf")
                z_lo, z_hi = float(zs.min()), float(zs.max())

                def _try_fit(A, mode, zb):
                    nonlocal best_err, best_coeffs, best_zb, best_mode
                    res, _, _, _ = np.linalg.lstsq(A, x_vals, rcond=None)
                    err = float(np.sum((A @ res - x_vals) ** 2))
                    if err < best_err:
                        best_err = err
                        best_coeffs = res
                        best_zb = zb
                        best_mode = mode

                # Pure straight: X = m*Z + n  (a=0)
                A_lin = np.column_stack([zs, np.ones_like(zs), np.zeros_like(zs)])
                _try_fit(A_lin, "straight", 0.0)
                straight_coeffs = best_coeffs.copy()
                straight_err = best_err

                # Pure curve: X = m*Z + n + a*Z^2
                A_quad = np.column_stack([zs, np.ones_like(zs), zs ** 2])
                _try_fit(A_quad, "curve", 0.0)

                # Hinge: ~20 breakpoint candidates
                for zb in np.linspace(z_lo, z_hi, 20):
                    for mode in ("straight_then_curve", "curve_then_straight"):
                        if mode == "straight_then_curve":
                            hinge = np.maximum(0, zs - zb) ** 2
                        else:
                            hinge = np.maximum(0, zb - zs) ** 2
                        A = np.column_stack([zs, np.ones_like(zs), hinge])
                        _try_fit(A, mode, zb)

                # Prefer straight if within 5% of the best fit
                if best_mode != "straight" and straight_err <= best_err * 1.05:
                    best_coeffs = straight_coeffs
                    best_mode = "straight"
                    best_zb = 0.0

                # Sample in image-Y space to guarantee edge-to-edge coverage
                for v in range(orig_h - 1, -1, -1):
                    w = _image_to_world(orig_w / 2, v, orig_w, orig_h, cam)
                    if w is None or w[1] <= 0:
                        continue
                    xw = _eval_hinge(w[1], best_coeffs, best_mode, best_zb)
                    uv = _world_to_image(float(xw), w[1], orig_w, orig_h, cam)
                    if uv and 0 <= uv[0] < orig_w and 0 <= uv[1] < orig_h:
                        centerline_img.append((int(round(uv[0])), int(round(uv[1]))))

    # Find the closest car whose bbox the centerline crosses through.
    # Evaluate the hinge model at each car's world Z to handle distant cars
    # whose bboxes are beyond the drawn centerline.
    lead_det = None
    best_y2 = -1
    bbox_extend_y = 5 * orig_h / _INPUT_H  # 5 model pixels downward
    if best_coeffs is not None and cam is not None:
        for det in detections:
            det_y2_ext = det["y2"] + bbox_extend_y
            # Sample several Y positions within the extended bbox
            for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
                check_y = det["y1"] + frac * (det_y2_ext - det["y1"])
                check_x = (det["x1"] + det["x2"]) / 2.0
                w = _image_to_world(check_x, check_y, orig_w, orig_h, cam)
                if w is None or w[1] <= 0:
                    continue
                xw = _eval_hinge(w[1], best_coeffs, best_mode, best_zb)
                uv = _world_to_image(float(xw), w[1], orig_w, orig_h, cam)
                if uv and det["x1"] <= uv[0] <= det["x2"] and det["y1"] <= uv[1] <= det_y2_ext:
                    if det["y2"] > best_y2:
                        best_y2 = det["y2"]
                        lead_det = det
                    break

    for det in detections:
        if det is lead_det:
            color = (0, 0, 255)
        else:
            color = (180, 180, 180)
        cv2.rectangle(result_bgr, (det["x1"], det["y1"]),
                      (det["x2"], det["y2"]), color, 2)

    if len(raw_centerline_img) >= 2:
        raw_arr = np.array(raw_centerline_img, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(result_bgr, [raw_arr], False, (0, 165, 255), 2, cv2.LINE_AA)
    if len(centerline_img) >= 2:
        pts_arr = np.array(centerline_img, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(result_bgr, [pts_arr], False, (0, 255, 255), 2, cv2.LINE_AA)

    _, jpeg = cv2.imencode(".jpg", result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jpeg.tobytes()


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
    finally:
        cap.release()

    jpeg_bytes = _render_debug(frame_bgr, Path(directory))
    return Response(content=jpeg_bytes, media_type="image/jpeg")


@router.get("/debug-images")
async def list_debug_images(directory: str):
    """List images in the Debug subdirectory (if it exists)."""
    debug_dir = Path(directory) / "Debug"
    if not debug_dir.is_dir():
        return {"images": []}
    images = sorted(
        f.name for f in debug_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    return {"images": images}


@router.get("/debug-image")
async def debug_image(filename: str, directory: str):
    """Run debug overlay on a standalone image from the Debug subdirectory."""
    path = Path(directory) / "Debug" / filename
    if not path.exists():
        raise HTTPException(404, f"Image not found: {path}")
    frame_bgr = cv2.imread(str(path))
    if frame_bgr is None:
        raise HTTPException(500, f"Cannot read image: {path}")
    jpeg_bytes = _render_debug(frame_bgr, Path(directory))
    return Response(content=jpeg_bytes, media_type="image/jpeg")


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
