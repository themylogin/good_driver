from __future__ import annotations

import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import httpx
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
from ..segmentation_mask import encode_mask, decode_mask, DRIVEABLE, LANE_ON_DRIVEABLE, LANE_NO_DRIVEABLE
from ..tracker import ByteTracker
from ..novatek_gps import extract_gps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/footage")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Thread pool — max_workers=1 keeps processing sequential
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=1)
_processing_videos: set[str] = set()   # "directory|filename" keys currently queued or running
_live_progress: dict[str, dict] = {}   # key → {"step": name, "processed_frames": N}

# ---------------------------------------------------------------------------
# Processing steps — ordered pipeline
# ---------------------------------------------------------------------------

PROCESSING_STEPS = [
    {"name": "inference", "label": "Inference"},
    {"name": "lead", "label": "Lead car"},
    {"name": "distances", "label": "Distances"},
    {"name": "gps", "label": "GPS"},
    {"name": "snap_to_road", "label": "Snap to road"},
]


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
    meta = json.loads(p.read_text())
    # Strip legacy keys
    meta.pop("processed_frames", None)
    meta.pop("steps", None)
    return meta


def _write_metadata(data_dir: Path, **fields) -> None:
    """Write/update metadata.json (video info only, no step progress)."""
    meta_path = data_dir / "metadata.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta.update(fields)
    meta.pop("steps", None)
    meta_path.write_text(json.dumps(meta))


def _count_inference_frames(ddir: Path) -> int:
    """Count how many frames have been inferred by checking batch files on disk."""
    count = 0
    for gz in ddir.rglob("*.json.gz"):
        # Batch files are under tiled dirs like 0/00/00.json.gz
        # Skip lead.json.gz, distances.json.gz, gps.json.gz etc.
        if gz.parent == ddir:
            continue
        try:
            data = json.loads(gzip.decompress(gz.read_bytes()))
            count += len(data)
        except Exception:
            pass
    return count


def _is_step_complete(directory: str, filename: str, step_name: str) -> bool:
    ddir = _data_dir(Path(directory), filename)
    meta = _read_metadata(ddir)
    if meta is None:
        return False
    total = meta.get("total_frames", 0)
    if total == 0:
        return False
    if step_name == "inference":
        # Check that the last batch file exists
        last_batch_start = ((total - 1) // 100) * 100
        return _frame_file_path(ddir, last_batch_start).exists()
    elif step_name == "lead":
        return (ddir / "lead.json.gz").exists()
    elif step_name == "distances":
        return (ddir / "distances.json.gz").exists()
    elif step_name == "gps":
        return (ddir / "gps.json.gz").exists()
    elif step_name == "snap_to_road":
        return (ddir / "snap_to_road.json.gz").exists()
    return False


def _all_steps_complete_from_dir(directory: str, filename: str) -> bool:
    return all(
        _is_step_complete(directory, filename, s["name"])
        for s in PROCESSING_STEPS
    )



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
    da_mask: np.ndarray, lane_mask: np.ndarray,
    orig_w: int, orig_h: int,
) -> tuple[list[np.ndarray], int | None, np.ndarray | None]:
    """Split driveable area by lane separators into individual lane masks.

    Returns (masks, ego_index, ego_discarded) where masks is a list of bool
    arrays at original resolution, ego_index is the index of the ego lane
    (or None), and ego_discarded is a bool mask of discarded ego-lane pixels
    (bottom crop + edge-touching rows) at original resolution.
    """
    da_cut = (da_mask & (1 - lane_mask)).astype(np.uint8)
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
            da_cut_no_bottom = (da_mask & (1 - lane_mask)).astype(np.uint8)
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

            # Re-split: trimming may have disconnected merged regions.
            # Keep only the sub-component closest to the bottom-center anchor.
            # Runs before top-crop so the crop doesn't sever bridge rows
            # that connect sub-components (wasting the crop budget).
            if comp.any():
                n_sub, sub_labels = cv2.connectedComponents(comp, connectivity=4)
                if n_sub > 2:  # more than just background + one component
                    best_sub = 0
                    best_bottom = -1
                    best_center_dist = float("inf")
                    anchor_col = _INPUT_W // 2
                    for s in range(1, n_sub):
                        sub_rows = np.where((sub_labels == s).any(axis=1))[0]
                        if len(sub_rows) == 0:
                            continue
                        bottom_row = int(sub_rows[-1])
                        cols = np.where(sub_labels[bottom_row] == s)[0]
                        center_dist = abs((cols[0] + cols[-1]) / 2 - anchor_col)
                        if bottom_row > best_bottom or (
                            bottom_row == best_bottom and center_dist < best_center_dist
                        ):
                            best_bottom = bottom_row
                            best_center_dist = center_dist
                            best_sub = s
                    comp = (sub_labels == best_sub).astype(np.uint8)

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

def _run_inference_step(filename: str, directory: str, key: str) -> None:
    """Step 1: Run YOLO inference + tracking on every frame."""
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

    # Write video info to metadata (no step progress)
    _write_metadata(ddir, total_frames=total,
                    fps=cap.get(cv2.CAP_PROP_FPS),
                    width=orig_w, height=orig_h)

    # Resume from last completed batch boundary (check files on disk)
    resume_frame = 0
    last_batch_start = ((total - 1) // 100) * 100
    for batch_start in range(0, total, 100):
        if _frame_file_path(ddir, batch_start).exists():
            resume_frame = min(batch_start + 100, total)
        else:
            break

    # Seek to resume position
    if resume_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
        logger.info("Resuming inference %s from frame %d/%d", filename, resume_frame, total)

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
            batch = []

        _live_progress[key] = {"step": "inference", "processed_frames": frame_n + 1}
        frame_n += 1

    # Flush remainder (< 100 frames)
    if batch:
        first_frame = frame_n - len(batch)
        fpath = _frame_file_path(ddir, first_frame)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(gzip.compress(json.dumps(batch).encode()))

    cap.release()
    logger.info("Finished inference for %s (%d frames)", filename, frame_n)


def _run_lead_step(filename: str, directory: str, key: str) -> None:
    """Step 2: Compute lead car (tracker_id + hwc distance) for every frame."""
    data_dir = Path(directory)
    ddir = _data_dir(data_dir, filename)
    meta = _read_metadata(ddir)
    if meta is None:
        logger.error("No metadata for %s, skipping lead step", filename)
        return

    total = meta.get("total_frames", 0)
    orig_w = meta.get("width", _INPUT_W)
    orig_h = meta.get("height", _INPUT_H)
    cam = _load_camera_params(data_dir)

    result: list[dict | None] = []
    frame_n = 0

    while frame_n < total:
        # Load batch
        batch_start = (frame_n // 100) * 100
        fpath = _frame_file_path(ddir, batch_start)
        if not fpath.exists():
            # Batch not available — fill with nulls
            batch_end = min(batch_start + 100, total)
            result.extend([None] * (batch_end - frame_n))
            frame_n = batch_end
            continue

        raw = fpath.read_bytes()
        try:
            data = gzip.decompress(raw) if fpath.suffix == ".gz" else raw
        except gzip.BadGzipFile:
            data = raw  # legacy uncompressed remainder
        frames = json.loads(data)

        # Process each frame in the batch
        offset = frame_n - batch_start
        for i in range(offset, len(frames)):
            frame_entry = frames[i]
            lead_entry: dict | None = None

            mask_payload = frame_entry.get("mask")
            detections = frame_entry.get("detections", [])

            if mask_payload is not None and cam is not None:
                classes_region = decode_mask(mask_payload)
                classes_full = np.zeros((_INPUT_H, _INPUT_W), dtype=np.uint8)
                start_row = mask_payload["start_row"]
                classes_full[start_row:start_row + mask_payload["row_count"]] = classes_region

                da_mask = ((classes_full == DRIVEABLE) | (classes_full == LANE_ON_DRIVEABLE)).astype(np.uint8)
                lane_mask = ((classes_full == LANE_ON_DRIVEABLE) | (classes_full == LANE_NO_DRIVEABLE)).astype(np.uint8)

                lane_masks, ego_idx, _ego_discarded = _build_lane_polygons(
                    da_mask, lane_mask, orig_w, orig_h)

                # Try ego-lane lead first, fall back to centerline lead
                if ego_idx is not None:
                    ego_mask = lane_masks[ego_idx]
                    ego_lead, hgp_dist = _find_ego_lane_lead(
                        detections, ego_mask, orig_w, orig_h, cam)
                    if ego_lead is not None and hgp_dist is not None:
                        lead_entry = {"id": ego_lead["track_id"], "distance": round(hgp_dist, 1)}

                if lead_entry is None:
                    centerline_img, centerline_extrap, _raw_cl, cl_lead = _fit_centerline_and_lead(
                        lane_masks, ego_idx, detections, orig_w, orig_h, cam)
                    if cl_lead is not None:
                        hwc_result = _compute_hwc_distance(
                            cl_lead, centerline_img, centerline_extrap, orig_w, cam)
                        if hwc_result is not None:
                            hwc_dist, _theta = hwc_result
                            lead_entry = {"id": cl_lead["track_id"], "distance": round(hwc_dist, 1)}

            result.append(lead_entry)
            frame_n += 1

            _live_progress[key] = {"step": "lead", "processed_frames": frame_n}

    # Write result atomically
    out_path = ddir / "lead.json.gz"
    tmp_path = ddir / "lead.json.gz.tmp"
    tmp_path.write_bytes(gzip.compress(json.dumps(result).encode()))
    tmp_path.rename(out_path)

    logger.info("Finished lead step for %s (%d frames)", filename, frame_n)


def _run_distances_step(filename: str, directory: str, key: str) -> None:
    """Step 3: Smooth lead data and write per-frame distances."""
    data_dir = Path(directory)
    ddir = _data_dir(data_dir, filename)
    meta = _read_metadata(ddir)
    if meta is None:
        logger.error("No metadata for %s, skipping distances step", filename)
        return

    total = meta.get("total_frames", 0)

    lead_path = ddir / "lead.json.gz"
    if not lead_path.exists():
        logger.error("No lead.json.gz for %s, skipping distances step", filename)
        return

    raw_lead = json.loads(gzip.decompress(lead_path.read_bytes()))
    smoothed = _smooth_lead_data(raw_lead)

    result: list[dict | None] = []
    for i, entry in enumerate(smoothed):
        if entry is not None:
            result.append({"tracker_id": entry["id"], "distance": entry["distance"]})
        else:
            result.append(None)
        if (i + 1) % 1000 == 0:
            _live_progress[key] = {"step": "distances", "processed_frames": i + 1}

    out_path = ddir / "distances.json.gz"
    tmp_path = ddir / "distances.json.gz.tmp"
    tmp_path.write_bytes(gzip.compress(json.dumps(result).encode()))
    tmp_path.rename(out_path)

    logger.info("Finished distances step for %s (%d frames)", filename, total)


def _run_gps_step(filename: str, directory: str, key: str) -> None:
    """Step 4: Extract GPS data from the video file."""
    data_dir = Path(directory)
    ddir = _data_dir(data_dir, filename)
    meta = _read_metadata(ddir)
    if meta is None:
        logger.error("No metadata for %s, skipping gps step", filename)
        return

    total = meta.get("total_frames", 0)

    video_path = data_dir / filename
    gps_data: list[dict | None] = []
    if video_path.exists():
        try:
            gps_data = extract_gps(video_path)
        except Exception:
            logger.exception("Failed to extract GPS from %s", filename)

    out_path = ddir / "gps.json.gz"
    tmp_path = ddir / "gps.json.gz.tmp"
    tmp_path.write_bytes(gzip.compress(json.dumps(gps_data).encode()))
    tmp_path.rename(out_path)

    logger.info("Finished gps step for %s (%d points)", filename, len(gps_data))


# ---------------------------------------------------------------------------
# Step 5 — Snap to road (OSRM map matching)
# ---------------------------------------------------------------------------

_SNAP_CHUNK_SIZE = 100
_SNAP_OVERLAP = 5
_SNAP_RADIUS = 50


def _load_settings(directory: str) -> dict:
    """Load settings.json from the directory, with defaults."""
    path = Path(directory) / "settings.json"
    defaults = {"osrm_url": "http://localhost:5000"}
    if path.exists():
        return {**defaults, **json.loads(path.read_text())}
    return defaults


def _osrm_match_chunk(client: httpx.Client, osrm_url: str, points: list[dict]) -> dict:
    """Call OSRM match API for a chunk of GPS points."""
    coords = ";".join(f"{p['lon']},{p['lat']}" for p in points)
    params: dict[str, str] = {
        "overview": "false",
        "geometries": "geojson",
        "steps": "true",
        "annotations": "true",
        "radiuses": ";".join([str(_SNAP_RADIUS)] * len(points)),
    }
    timestamps = [p.get("ts") for p in points]
    if all(t is not None for t in timestamps):
        params["timestamps"] = ";".join(str(t) for t in timestamps)

    resp = client.get(f"{osrm_url.rstrip('/')}/match/v1/driving/{coords}", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _run_snap_to_road_step(filename: str, directory: str, key: str) -> None:
    """Step 5: Snap GPS track to road network via OSRM map matching."""
    data_dir = Path(directory)
    ddir = _data_dir(data_dir, filename)
    meta = _read_metadata(ddir)
    if meta is None:
        logger.error("No metadata for %s, skipping snap_to_road step", filename)
        return

    total = meta.get("total_frames", 0)

    # Load raw GPS data from previous step
    gps_path = ddir / "gps.json.gz"
    if not gps_path.exists():
        logger.error("No gps.json.gz for %s, skipping snap_to_road step", filename)
        return

    gps_data: list[dict | None] = json.loads(gzip.decompress(gps_path.read_bytes()))
    if not gps_data:
        # No GPS points — write empty result
        out_path = ddir / "snap_to_road.json.gz"
        tmp_path = ddir / "snap_to_road.json.gz.tmp"
        tmp_path.write_bytes(gzip.compress(json.dumps([]).encode()))
        tmp_path.rename(out_path)
        logger.info("No GPS data for %s, wrote empty snap_to_road", filename)
        return

    settings = _load_settings(directory)
    osrm_url = settings.get("osrm_url", "http://localhost:5000")

    # Convert GPS entries to format for OSRM (lat, lon, unix timestamp).
    # gps_data may contain None entries (no GPS fix for that second), so we
    # build a compact list of only valid points and keep a mapping back to
    # the original (full-length) indices.
    points: list[dict] = []
    valid_to_orig: list[int] = []       # points[i] came from gps_data[valid_to_orig[i]]
    for orig_idx, g in enumerate(gps_data):
        if g is None:
            continue
        ts = None
        if g.get("datetime"):
            try:
                dt = datetime.fromisoformat(g["datetime"].replace("Z", "+00:00"))
                ts = int(dt.timestamp())
            except (ValueError, TypeError):
                pass
        points.append({"lat": g["lat"], "lon": g["lon"], "ts": ts})
        valid_to_orig.append(orig_idx)

    if not points:
        # All GPS entries are None (no fix at all) — write all-None result
        result = [None] * len(gps_data)
        out_path = ddir / "snap_to_road.json.gz"
        tmp_path = ddir / "snap_to_road.json.gz.tmp"
        tmp_path.write_bytes(gzip.compress(json.dumps(result).encode()))
        tmp_path.rename(out_path)
        logger.info("No valid GPS fixes for %s, wrote all-None snap_to_road", filename)
        return

    # Call OSRM in chunks
    wp_lookup: dict[tuple[int, int], dict] = {}
    all_matchings: list[dict] = []
    all_tracepoints: list[dict | None] = []

    step = _SNAP_CHUNK_SIZE - _SNAP_OVERLAP
    chunk_num = -(-len(points) // step)

    with httpx.Client() as client:
        for i, start in enumerate(range(0, len(points), step)):
            chunk = points[start : start + _SNAP_CHUNK_SIZE]

            _live_progress[key] = {
                "step": "snap_to_road",
                "processed_frames": int((i + 1) / chunk_num * total) if chunk_num else total,
            }

            try:
                data = _osrm_match_chunk(client, osrm_url, chunk)
            except Exception:
                logger.exception("OSRM match failed for chunk %d of %s", i, filename)
                data = {"code": "Error"}

            if data.get("code") != "Ok":
                logger.warning("OSRM: %s — %s (chunk %d of %s)",
                               data.get("code"), data.get("message", ""), i, filename)
                tracepoints: list[dict | None] = [None] * len(chunk)
                matchings: list[dict] = []
            else:
                offset = len(all_matchings)
                tracepoints = data["tracepoints"]
                for tp in tracepoints:
                    if tp is not None:
                        tp["matchings_index"] += offset
                matchings = data["matchings"]

            # Build waypoint lookup
            for j, tp in enumerate(tracepoints):
                if tp is None:
                    continue
                k = (tp["matchings_index"], tp["waypoint_index"])
                if k not in wp_lookup:
                    lon, lat = tp["location"]
                    orig_idx = valid_to_orig[start + j]
                    wp_lookup[k] = {
                        "lat": lat,
                        "lon": lon,
                        "datetime": gps_data[orig_idx].get("datetime"),
                        "road_name": tp.get("name") or None,
                        "orig_idx": orig_idx,
                    }

            if all_tracepoints:
                tracepoints = tracepoints[_SNAP_OVERLAP:]
            all_tracepoints.extend(tracepoints)
            all_matchings.extend(matchings)

    # Build per-tracepoint snapped data indexed by original GPS index
    # tracepoint → (matching_idx, waypoint_idx) → lookup entry
    snapped_by_orig: dict[int, dict] = {}
    for tp in all_tracepoints:
        if tp is None:
            continue
        k = (tp["matchings_index"], tp["waypoint_index"])
        entry = wp_lookup.get(k)
        if entry is not None:
            snapped_by_orig[entry["orig_idx"]] = entry

    # For speed limit, build from legs (between consecutive waypoints)
    leg_data_by_orig: dict[int, dict] = {}
    for mi, matching in enumerate(all_matchings):
        for li, leg in enumerate(matching.get("legs", [])):
            wp_from = wp_lookup.get((mi, li))
            if wp_from is None:
                continue

            speed_limit_kmh = None
            steps = leg.get("steps", [])
            if steps:
                ref = steps[0].get("ref", "")
                if ref:
                    try:
                        speed_limit_kmh = int(ref)
                    except ValueError:
                        pass

            leg_data_by_orig[wp_from["orig_idx"]] = {
                "speed_limit_kmh": speed_limit_kmh,
            }

    # Build final per-second result aligned to gps_data indices
    result: list[dict | None] = []
    for idx, g in enumerate(gps_data):
        if g is None:
            result.append(None)
            continue
        snapped = snapped_by_orig.get(idx)
        leg = leg_data_by_orig.get(idx)
        if snapped is not None:
            result.append({
                "timestamp": g.get("datetime"),
                "lat": snapped["lat"],
                "lon": snapped["lon"],
                "speed_limit_kmh": leg["speed_limit_kmh"] if leg else None,
                "road_name": snapped.get("road_name"),
            })
        else:
            result.append(None)

    out_path = ddir / "snap_to_road.json.gz"
    tmp_path = ddir / "snap_to_road.json.gz.tmp"
    tmp_path.write_bytes(gzip.compress(json.dumps(result).encode()))
    tmp_path.rename(out_path)

    matched = sum(1 for r in result if r is not None)
    logger.info("Finished snap_to_road step for %s (%d/%d matched)", filename, matched, len(result))


def _process_video_worker(filename: str, directory: str, key: str) -> None:
    """Run all processing steps sequentially for one video."""
    try:
        for step in PROCESSING_STEPS:
            if _is_step_complete(directory, filename, step["name"]):
                continue
            if step["name"] == "inference":
                _run_inference_step(filename, directory, key)
            elif step["name"] == "lead":
                _run_lead_step(filename, directory, key)
            elif step["name"] == "distances":
                _run_distances_step(filename, directory, key)
            elif step["name"] == "gps":
                _run_gps_step(filename, directory, key)
            elif step["name"] == "snap_to_road":
                _run_snap_to_road_step(filename, directory, key)
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
        return {"total_frames": 0, "steps": {}}
    total = meta.get("total_frames", 0)

    # Build steps from file existence
    steps: dict[str, dict] = {}
    for s in PROCESSING_STEPS:
        name = s["name"]
        if _is_step_complete(directory, filename, name):
            steps[name] = {"processed_frames": total}
        else:
            steps[name] = {"processed_frames": 0}

    # Overlay live progress for the currently running step
    key = f"{directory}|{filename}"
    live = _live_progress.get(key)
    current_step = None
    if live is not None:
        step_name = live["step"]
        current_step = step_name
        step_info = steps.setdefault(step_name, {})
        if live["processed_frames"] > step_info.get("processed_frames", 0):
            step_info["processed_frames"] = live["processed_frames"]

    meta["steps"] = steps
    meta["current_step"] = current_step
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


_LEAD_PALETTE = [
    (0, 255, 0),    (255, 0, 0),    (0, 128, 255),  (255, 255, 0),
    (255, 0, 255),  (0, 255, 255),  (255, 128, 0),  (128, 0, 255),
    (0, 255, 128),  (255, 0, 128),  (128, 255, 0),  (0, 128, 128),
    (255, 128, 128),(128, 128, 255),(128, 255, 255),(255, 255, 128),
]


def _load_lead_data(directory: str, filename: str) -> list[dict | None]:
    ddir = _data_dir(Path(directory), filename)
    lead_path = ddir / "lead.json.gz"
    if not lead_path.exists():
        raise HTTPException(404, "Lead step not completed")
    data = json.loads(gzip.decompress(lead_path.read_bytes()))
    if len(data) == 0:
        raise HTTPException(404, "No frames")
    return data


def _render_lead_timeline(lead_data: list[dict | None]) -> bytes:
    total = len(lead_data)
    h = 50
    img = np.zeros((h, total, 3), dtype=np.uint8)
    max_distance = 100.0

    # Build stable color mapping: assign palette index by first-appearance order
    seen_ids: list[int] = []
    for entry in lead_data:
        if entry is not None and entry["id"] not in seen_ids:
            seen_ids.append(entry["id"])
    id_to_color = {tid: idx % len(_LEAD_PALETTE) for idx, tid in enumerate(seen_ids)}

    for x, entry in enumerate(lead_data):
        if entry is None:
            continue
        dist = min(entry["distance"], max_distance)
        bar_h = max(1, int(dist / max_distance * (h / 2)))
        color = _LEAD_PALETTE[id_to_color[entry["id"]]]
        img[h - bar_h:h, x] = color

    _, png = cv2.imencode(".png", img)
    return png.tobytes()


def _smooth_lead_data(lead_data: list[dict | None], kernel: int = 15) -> list[dict | None]:
    """Smooth lead data: remove short intrusions, interpolate gaps, median-filter."""
    from scipy.ndimage import median_filter

    n = len(lead_data)
    ids = np.full(n, -1, dtype=np.int32)
    dists = np.full(n, np.nan, dtype=np.float64)

    for i, entry in enumerate(lead_data):
        if entry is not None:
            ids[i] = entry["id"]
            dists[i] = entry["distance"]

    # Erase short intrusions: if tracker B appears for <=30 frames between
    # runs of tracker A (or nulls), replace B with -1 so gap-fill can bridge.
    _MAX_INTRUSION = 30
    i = 0
    while i < n:
        if ids[i] < 0:
            i += 1
            continue
        seg_id = ids[i]
        j = i
        while j < n and ids[j] == seg_id:
            j += 1
        seg_len = j - i
        if seg_len <= _MAX_INTRUSION:
            # Look past nulls to find actual surrounding tracker ids
            prev_id = -1
            for k in range(i - 1, -1, -1):
                if ids[k] >= 0:
                    prev_id = ids[k]
                    break
            next_id = -1
            for k in range(j, n):
                if ids[k] >= 0:
                    next_id = ids[k]
                    break
            # Erase if surrounded by a different id on both sides
            if prev_id != seg_id and next_id != seg_id:
                ids[i:j] = -1
                dists[i:j] = np.nan
        i = j

    # Interpolate gaps up to 100 frames between same tracker id
    _MAX_GAP = 100
    last_id = -1
    last_dist = np.nan
    gap_start = -1
    for i in range(n):
        if ids[i] >= 0:
            if gap_start >= 0 and ids[i] == last_id and (i - gap_start) <= _MAX_GAP:
                gap_len = i - gap_start
                for j in range(gap_start, i):
                    t = (j - gap_start + 1) / (gap_len + 1)
                    ids[j] = last_id
                    dists[j] = last_dist + t * (dists[i] - last_dist)
            last_id = ids[i]
            last_dist = dists[i]
            gap_start = -1
        else:
            if gap_start < 0:
                gap_start = i

    # Median-filter distance per contiguous tracker segment
    result: list[dict | None] = [None] * n
    i = 0
    while i < n:
        if ids[i] < 0:
            i += 1
            continue
        seg_id = ids[i]
        j = i
        while j < n and ids[j] == seg_id:
            j += 1
        seg_dists = dists[i:j].copy()
        if len(seg_dists) >= kernel:
            seg_dists = median_filter(seg_dists, size=kernel)
        for k in range(i, j):
            result[k] = {"id": int(seg_id), "distance": round(float(seg_dists[k - i]), 1)}
        i = j

    return result


@router.get("/lead-timeline")
async def lead_timeline(filename: str, directory: str):
    """Return a PNG timeline of raw lead-car distance."""
    lead_data = _load_lead_data(directory, filename)
    return Response(
        content=_render_lead_timeline(lead_data),
        media_type="image/png",
    )


@router.get("/lead-timeline-smooth")
async def lead_timeline_smooth(filename: str, directory: str):
    """Return a PNG timeline of smoothed lead-car distance."""
    # Prefer pre-computed data from distances step
    ddir = _data_dir(Path(directory), filename)
    dist_path = ddir / "distances.json.gz"
    if dist_path.exists():
        entries = json.loads(gzip.decompress(dist_path.read_bytes()))
        smoothed: list[dict | None] = []
        for entry in entries:
            if entry is not None:
                smoothed.append({"id": entry["tracker_id"], "distance": entry["distance"]})
            else:
                smoothed.append(None)
        return Response(
            content=_render_lead_timeline(smoothed),
            media_type="image/png",
        )
    # Fallback: compute on the fly
    lead_data = _load_lead_data(directory, filename)
    smoothed = _smooth_lead_data(lead_data)
    return Response(
        content=_render_lead_timeline(smoothed),
        media_type="image/png",
    )


@router.get("/gps-info")
async def gps_info(filename: str, directory: str, frame: int):
    """Return GPS info for a given frame."""
    ddir = _data_dir(Path(directory), filename)
    gps_path = ddir / "gps.json.gz"
    if not gps_path.exists():
        raise HTTPException(404, "GPS step not completed")

    meta = _read_metadata(ddir)
    video_fps = meta.get("fps", 30.0) if meta else 30.0
    block_size = round(video_fps)

    gps_data = json.loads(gzip.decompress(gps_path.read_bytes()))
    gps_idx = frame // block_size

    gps = gps_data[gps_idx] if gps_idx < len(gps_data) else None
    return {"gps": gps}


@router.get("/snap-to-road-info")
async def snap_to_road_info(filename: str, directory: str, frame: int):
    """Return snapped-to-road info for a given frame."""
    ddir = _data_dir(Path(directory), filename)
    snap_path = ddir / "snap_to_road.json.gz"
    if not snap_path.exists():
        raise HTTPException(404, "Snap to road step not completed")

    meta = _read_metadata(ddir)
    video_fps = meta.get("fps", 30.0) if meta else 30.0
    block_size = round(video_fps)

    snap_data = json.loads(gzip.decompress(snap_path.read_bytes()))
    idx = frame // block_size

    snap = snap_data[idx] if idx < len(snap_data) else None
    return {"snap": snap}


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


def _fit_centerline_and_lead(
    lane_masks: list[np.ndarray],
    ego_idx: int | None,
    detections: list[dict],
    orig_w: int, orig_h: int,
    cam: dict | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], dict | None]:
    """Compute fitted centerline and identify lead car.

    Returns (centerline_img, centerline_extrap, raw_centerline_img, lead_det).
    """
    centerline_img: list[tuple[int, int]] = []
    centerline_extrap: list[tuple[int, int]] = []
    raw_centerline_img: list[tuple[int, int]] = []
    best_coeffs = None
    best_zb = 0.0
    best_mode = "straight_then_curve"

    if ego_idx is not None and ego_idx < len(lane_masks) and cam is not None:
        lane_polys_world, _ = _project_lane_boundaries(lane_masks, orig_w, orig_h, cam)
        lane_interps = _build_lane_interps(lane_polys_world)
        if ego_idx < len(lane_interps):
            interp = lane_interps[ego_idx]
        else:
            interp = None
        if interp is not None:
            zl, xl, zr, xr = interp
            z_min = max(zl[0], zr[0])
            z_max = min(zl[-1], zr[-1])
            z_samples = np.linspace(z_min, z_max, 100)
            cx_raw = np.array([
                (np.interp(z, zl, xl) + np.interp(z, zr, xr)) / 2.0
                for z in z_samples
            ])
            _dx = np.diff(cx_raw)
            _dz = np.diff(z_samples)
            _lat_vel = np.abs(_dx / _dz)
            _med_vel = max(float(np.median(_lat_vel)), 0.01)
            _thresh = max(_LATERAL_VEL_OUTLIER_MULT * _med_vel, _LATERAL_VEL_MIN_THRESH)
            # Mark points adjacent to lateral velocity spikes as outliers
            _good = np.ones(len(cx_raw), dtype=bool)
            for _i in range(len(_lat_vel)):
                if _lat_vel[_i] > _thresh:
                    _good[_i] = False
                    _good[_i + 1] = False
            cx_trimmed = cx_raw[_good]
            z_trimmed = z_samples[_good]
            # Raw centerline image points
            for i, z in enumerate(z_samples):
                uv = _world_to_image(float(cx_raw[i]), float(z), orig_w, orig_h, cam)
                if uv and 0 <= uv[0] < orig_w and 0 <= uv[1] < orig_h:
                    raw_centerline_img.append((int(round(uv[0])), int(round(uv[1]))))
            # Trimmed centers for hinge fit
            world_centers: list[tuple[float, float]] = []
            for i, z in enumerate(z_trimmed):
                world_centers.append((float(cx_trimmed[i]), float(z)))
            # Fitted centerline: linear+parabolic hinge model
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

                A_lin = np.column_stack([zs, np.ones_like(zs), np.zeros_like(zs)])
                _try_fit(A_lin, "straight", 0.0)
                straight_coeffs = best_coeffs.copy()
                straight_err = best_err

                A_quad = np.column_stack([zs, np.ones_like(zs), zs ** 2])
                _try_fit(A_quad, "curve", 0.0)

                for zb in np.linspace(z_lo, z_hi, 20):
                    for mode in ("straight_then_curve", "curve_then_straight"):
                        if mode == "straight_then_curve":
                            hinge = np.maximum(0, zs - zb) ** 2
                        else:
                            hinge = np.maximum(0, zb - zs) ** 2
                        A = np.column_stack([zs, np.ones_like(zs), hinge])
                        _try_fit(A, mode, zb)

                if best_mode != "straight" and straight_err <= best_err * 1.05:
                    best_coeffs = straight_coeffs
                    best_mode = "straight"
                    best_zb = 0.0

                for v in range(orig_h - 1, -1, -1):
                    w = _image_to_world(orig_w / 2, v, orig_w, orig_h, cam)
                    if w is None or w[1] <= 0:
                        continue
                    xw = _eval_hinge(w[1], best_coeffs, best_mode, best_zb)
                    uv = _world_to_image(float(xw), w[1], orig_w, orig_h, cam)
                    if uv and 0 <= uv[0] < orig_w and 0 <= uv[1] < orig_h:
                        centerline_img.append((int(round(uv[0])), int(round(uv[1]))))

                # Extrapolate past the horizon in image space
                if len(centerline_img) >= 10:
                    tail = np.array(centerline_img[-10:], dtype=np.float64)
                    xs, ys = tail[:, 0], tail[:, 1]
                    coeffs = np.polyfit(ys, xs, 1)
                    last_x, last_y = centerline_img[-1]
                    centerline_extrap.append((last_x, last_y))  # overlap point
                    for ey in range(last_y - 1, -1, -1):
                        ex = int(round(coeffs[0] * ey + coeffs[1]))
                        if 0 <= ex < orig_w:
                            centerline_extrap.append((ex, ey))
                        else:
                            break

    # Find the first car the centerline crosses, walking bottom-to-top.
    lead_det = None
    bbox_extend_y = 0
    ego_mask = lane_masks[ego_idx] if ego_idx is not None and ego_idx < len(lane_masks) else None
    skip_dets: set[int] = set()
    all_cl = [*centerline_img, *centerline_extrap]
    for ci, (cx_px, cy_px) in enumerate(all_cl):
        for di, det in enumerate(detections):
            if di in skip_dets:
                continue
            if (det["x1"] <= cx_px <= det["x2"]
                    and det["y1"] <= cy_px <= det["y2"] + bbox_extend_y):
                # Skip cars in another lane: if the centerline enters via
                # a vertical (side) bbox edge and the ego lane extends 2+ px
                # above the car's bottom, the car isn't in our lane.
                if ci > 0 and ego_mask is not None:
                    # Exact edge crossing: which bbox edge does the segment
                    # (prev_point -> this_point) cross first?
                    px, py = all_cl[ci - 1]
                    dx, dy = cx_px - px, cy_px - py
                    entered_side = False
                    best_t = 2.0
                    # Left edge (x = x1)
                    if dx != 0:
                        t = (det["x1"] - px) / dx
                        if 0 <= t <= 1:
                            yt = py + t * dy
                            if det["y1"] <= yt <= det["y2"] + bbox_extend_y and t < best_t:
                                best_t, entered_side = t, True
                    # Right edge (x = x2)
                    if dx != 0:
                        t = (det["x2"] - px) / dx
                        if 0 <= t <= 1:
                            yt = py + t * dy
                            if det["y1"] <= yt <= det["y2"] + bbox_extend_y and t < best_t:
                                best_t, entered_side = t, True
                    # Bottom edge (y = y2 + extend)
                    if dy != 0:
                        t = (det["y2"] + bbox_extend_y - py) / dy
                        if 0 <= t <= 1:
                            xt = px + t * dx
                            if det["x1"] <= xt <= det["x2"] and t < best_t:
                                best_t, entered_side = t, False
                    # Top edge (y = y1)
                    if dy != 0:
                        t = (det["y1"] - py) / dy
                        if 0 <= t <= 1:
                            xt = px + t * dx
                            if det["x1"] <= xt <= det["x2"] and t < best_t:
                                best_t, entered_side = t, False

                    if entered_side:
                        check_y = det["y2"] - 2
                        if 0 <= check_y < orig_h and ego_mask[:check_y + 1, :].any():
                            skip_dets.add(di)
                            continue
                lead_det = det
                break
        if lead_det is not None:
            break

    return centerline_img, centerline_extrap, raw_centerline_img, lead_det


def _compute_hwc_distance(
    lead_det: dict,
    centerline_img: list[tuple[int, int]],
    centerline_extrap: list[tuple[int, int]],
    orig_w: int,
    cam: dict,
) -> tuple[float, float] | None:
    """Compute angle-corrected apparent-width distance to a lead car.

    Returns (hwc_distance, theta_degrees), or None if it can't be computed.
    """
    fx_scaled = cam["fx"] * (orig_w / cam["image_width"])
    bbox_w_px = lead_det["x2"] - lead_det["x1"]
    if bbox_w_px <= 0:
        return None
    hw_dist = 2.0 * fx_scaled / bbox_w_px

    all_cl = [*centerline_img, *centerline_extrap]
    entry_idx = None
    for ci, (cpx, cpy) in enumerate(all_cl):
        if (lead_det["x1"] <= cpx <= lead_det["x2"]
                and lead_det["y1"] <= cpy <= lead_det["y2"]):
            entry_idx = ci
            break
    if entry_idx is None or len(all_cl) < 2:
        return None

    lo = max(0, entry_idx - 5)
    hi = min(len(all_cl), entry_idx + 6)
    if hi - lo < 2:
        return None

    seg = np.array(all_cl[lo:hi], dtype=np.float64)
    dx = seg[-1, 0] - seg[0, 0]
    dy = seg[-1, 1] - seg[0, 1]
    theta_deg = math.degrees(math.atan2(abs(dy), abs(dx)))
    f_theta = float(np.interp(
        theta_deg, [0, 13, 24, 40, 90], [0.60, 0.73, 0.86, 0.91, 1.0]
    ))
    f_theta = max(0.01, f_theta)
    return hw_dist / f_theta, theta_deg


def _find_ego_lane_lead(
    detections: list[dict],
    ego_mask: np.ndarray,
    orig_w: int, orig_h: int,
    cam: dict | None,
) -> tuple[dict | None, float | None]:
    """Find the lead car using ego-lane proximity criteria and compute hGP distance.

    Criteria (thresholds in model pixels):
      1. Non-discarded ego lane pixels within 5 px below bbox bottom
      2. Ego lane pixels within 10 px left of left edge OR right of right edge
      3. 90% of bbox height is above some ego lane pixel (ego pixels in bottom 10% of bbox)

    Returns (detection, hgp_distance_m) or (None, None).
    """
    if cam is None:
        return None, None

    sy = orig_h / _INPUT_H
    sx = orig_w / _INPUT_W
    below_thresh = int(round(5 * sy))
    side_thresh = int(round(10 * sx))

    candidates: list[tuple[dict, float]] = []
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        bbox_h = y2 - y1
        if bbox_h <= 0:
            continue

        # Condition 1: ego lane pixels within 5 model px below bbox bottom
        by0 = min(y2, orig_h)
        by1 = min(y2 + below_thresh, orig_h)
        bx0 = max(x1, 0)
        bx1 = min(x2, orig_w)
        if by0 >= by1 or bx0 >= bx1:
            continue
        if not ego_mask[by0:by1, bx0:bx1].any():
            continue

        # Condition 2: ego lane pixels within 10 model px to left or right
        ey0 = max(y1, 0)
        ey1 = min(y2, orig_h)
        if ey0 >= ey1:
            continue
        has_side = False
        lx0, lx1 = max(x1 - side_thresh, 0), max(x1, 0)
        if lx0 < lx1 and ego_mask[ey0:ey1, lx0:lx1].any():
            has_side = True
        if not has_side:
            rx0, rx1 = min(x2, orig_w), min(x2 + side_thresh, orig_w)
            if rx0 < rx1 and ego_mask[ey0:ey1, rx0:rx1].any():
                has_side = True
        if not has_side:
            continue

        # Condition 3: 90% of bbox columns have ego lane pixels somewhere below them
        bbox_cols = bx1 - bx0
        if bbox_cols <= 0:
            continue
        ego_below = ego_mask[by0:, bx0:bx1]  # from bbox bottom to image bottom
        cols_with_ego = int((ego_below.any(axis=0)).sum())
        if cols_with_ego / bbox_cols < 0.9:
            continue

        # Compute hGP distance — discard if > 25m (fall back to centerline)
        cx = (x1 + x2) / 2.0
        w = _image_to_world(cx, y2, orig_w, orig_h, cam)
        if w is None or w[1] <= 0 or w[1] > 25:
            continue
        candidates.append((det, w[1]))

    if not candidates:
        return None, None

    # Closest candidate (smallest hGP)
    candidates.sort(key=lambda c: c[1])
    return candidates[0]


def _draw_overlays(
    canvas: np.ndarray,
    lane_masks: list[np.ndarray],
    ego_idx: int | None,
    ego_discarded: np.ndarray | None,
    lane_mask: np.ndarray,
    detections: list[dict],
    lead_det: dict | None,
    centerline_img: list[tuple[int, int]],
    centerline_extrap: list[tuple[int, int]],
    raw_centerline_img: list[tuple[int, int]],
    orig_w: int, orig_h: int,
    cam: dict | None = None,
    hgp_distance: float | None = None,
) -> None:
    """Draw all overlay elements onto a BGRA canvas (mutates in-place)."""
    alpha = int(0.4 * 255)

    # Lane polygons
    if ego_idx is not None:
        ego_full = lane_masks[ego_idx]
        canvas[ego_full] = [0, 255, 0, alpha]
        if ego_discarded is not None and ego_discarded.any():
            canvas[ego_discarded] = [0, 255, 0, int(0.12 * 255)]
        for i, lm in enumerate(lane_masks):
            if i != ego_idx and lm.any():
                canvas[lm] = [255, 100, 0, alpha]

    # Lane separators
    ll_full = cv2.resize(lane_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    lanes = ll_full > 0
    if lanes.any():
        canvas[lanes] = [0, 0, 255, alpha]

    # Bounding boxes — draw non-lead first, then lead on top
    for det in detections:
        if det is lead_det:
            continue
        cv2.rectangle(canvas, (det["x1"], det["y1"]),
                      (det["x2"], det["y2"]), (180, 180, 180, 255), 2)

    if lead_det is not None:
        cv2.rectangle(canvas, (lead_det["x1"], lead_det["y1"]),
                      (lead_det["x2"], lead_det["y2"]), (0, 0, 255, 255), 2)
        if hgp_distance is not None:
            # Ego-lane lead: simple h= label
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.5, orig_h / 1080) * 0.6
            thickness = max(1, int(orig_h / 540 * 0.6 + 1))
            label = f"h= {hgp_distance:.0f}m"
            bbox_cx = lead_det["x1"] + (lead_det["x2"] - lead_det["x1"]) // 2
            bot_y = lead_det["y1"] - 6
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = bbox_cx - tw // 2
            cv2.rectangle(canvas, (tx - 2, bot_y - th - 2),
                          (tx + tw + 2, bot_y + 2), (0, 0, 0, 200), -1)
            cv2.putText(canvas, label, (tx, bot_y), font, scale,
                        (255, 255, 255, 255), thickness, cv2.LINE_AA)
        elif cam is not None:
            # Centerline-based lead: full labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.5, orig_h / 1080) * 0.6
            thickness = max(1, int(orig_h / 540 * 0.6 + 1))
            lines: list[str] = []
            hwc_label: str | None = None

            fx_scaled = cam["fx"] * (orig_w / cam["image_width"])

            cx = (lead_det["x1"] + lead_det["x2"]) / 2.0
            w = _image_to_world(cx, lead_det["y2"], orig_w, orig_h, cam)
            if w is not None and w[1] > 0:
                lines.append(f"hGP: {w[1]:.0f}m")

            bbox_h_px = lead_det["y2"] - lead_det["y1"]
            if bbox_h_px > 0:
                lines.append(f"hh: {1.5 * fx_scaled / bbox_h_px:.0f}m")

            bbox_w_px = lead_det["x2"] - lead_det["x1"]
            if bbox_w_px > 0:
                hw_dist = 2.0 * fx_scaled / bbox_w_px
                lines.append(f"hw: {hw_dist:.0f}m")

                hwc_result = _compute_hwc_distance(lead_det, centerline_img, centerline_extrap, orig_w, cam)
                if hwc_result is not None:
                    hwc_dist, theta_deg = hwc_result
                    hwc_label = f"a: {theta_deg:.0f} hwc: {hwc_dist:.0f}m"

            if lines or hwc_label:
                bbox_cx = lead_det["x1"] + (lead_det["x2"] - lead_det["x1"]) // 2
                bot_y = lead_det["y1"] - 6
                if hwc_label:
                    (tw2, th2), _ = cv2.getTextSize(hwc_label, font, scale, thickness)
                    tx2 = bbox_cx - tw2 // 2
                    cv2.rectangle(canvas, (tx2 - 2, bot_y - th2 - 2),
                                  (tx2 + tw2 + 2, bot_y + 2), (0, 0, 0, 200), -1)
                    cv2.putText(canvas, hwc_label, (tx2, bot_y), font, scale,
                                (255, 255, 255, 255), thickness, cv2.LINE_AA)
                    bot_y -= th2 + 4
                if lines:
                    label = "  ".join(lines)
                    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
                    tx = bbox_cx - tw // 2
                    cv2.rectangle(canvas, (tx - 2, bot_y - th - 2),
                                  (tx + tw + 2, bot_y + 2), (0, 0, 0, 200), -1)
                    cv2.putText(canvas, label, (tx, bot_y), font, scale,
                                (255, 255, 255, 255), thickness, cv2.LINE_AA)

    # Centerlines — skip when ego-lane lead is found (centerline not used for that)
    if hgp_distance is None:
        if len(raw_centerline_img) >= 2:
            raw_arr = np.array(raw_centerline_img, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [raw_arr], False, (0, 165, 255, 255), 2, cv2.LINE_AA)
        if len(centerline_img) >= 2:
            pts_arr = np.array(centerline_img, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_arr], False, (0, 255, 255, 255), 2, cv2.LINE_AA)
        # Extrapolated centerline as dashes
        if len(centerline_extrap) >= 2:
            _DASH = 12
            _GAP = 8
            for i in range(0, len(centerline_extrap) - 1, _DASH + _GAP):
                seg = centerline_extrap[i : i + _DASH + 1]
                if len(seg) >= 2:
                    seg_arr = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [seg_arr], False, (0, 255, 255, 255), 2, cv2.LINE_AA)


def _render_overlay_png(
    da_mask: np.ndarray,
    lane_mask: np.ndarray,
    detections: list[dict],
    orig_w: int, orig_h: int,
    cam: dict | None,
) -> bytes:
    """Render overlay on transparent BGRA canvas and return PNG bytes."""
    lane_masks, ego_idx, ego_discarded = _build_lane_polygons(
        da_mask, lane_mask, orig_w, orig_h)

    centerline_img, centerline_extrap, raw_centerline_img, cl_lead = _fit_centerline_and_lead(
        lane_masks, ego_idx, detections, orig_w, orig_h, cam)

    ego_mask = lane_masks[ego_idx] if ego_idx is not None else np.zeros((orig_h, orig_w), dtype=bool)
    ego_lead, hgp_dist = _find_ego_lane_lead(detections, ego_mask, orig_w, orig_h, cam)
    lead_det = ego_lead if ego_lead is not None else cl_lead

    canvas = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    _draw_overlays(canvas, lane_masks, ego_idx, ego_discarded, lane_mask,
                   detections, lead_det, centerline_img, centerline_extrap,
                   raw_centerline_img, orig_w, orig_h, cam=cam, hgp_distance=hgp_dist)

    _, png = cv2.imencode(".png", canvas)
    return png.tobytes()


def _render_debug(frame_bgr: np.ndarray, cam_directory: Path) -> bytes:
    """Render debug overlay on a BGR frame and return JPEG bytes."""
    orig_h, orig_w = frame_bgr.shape[:2]

    session = _get_session()
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor, sx, sy = _preprocess(img)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})

    detections = _extract_detections(outputs, sx, sy, orig_w, orig_h)
    cam = _load_camera_params(cam_directory)

    masks_result = _extract_masks(outputs)
    if masks_result is not None:
        da_mask, lane_mask_model = masks_result
    else:
        da_mask = np.zeros((_INPUT_H, _INPUT_W), dtype=np.uint8)
        lane_mask_model = np.zeros((_INPUT_H, _INPUT_W), dtype=np.uint8)

    # Render transparent overlay then composite onto the video frame
    overlay_bgra = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    lane_masks, ego_idx, ego_discarded = _build_lane_polygons(
        da_mask, lane_mask_model, orig_w, orig_h)

    centerline_img, centerline_extrap, raw_centerline_img, cl_lead = _fit_centerline_and_lead(
        lane_masks, ego_idx, detections, orig_w, orig_h, cam)

    ego_mask = lane_masks[ego_idx] if ego_idx is not None else np.zeros((orig_h, orig_w), dtype=bool)
    ego_lead, hgp_dist = _find_ego_lane_lead(detections, ego_mask, orig_w, orig_h, cam)
    lead_det = ego_lead if ego_lead is not None else cl_lead

    _draw_overlays(overlay_bgra, lane_masks, ego_idx, ego_discarded, lane_mask_model,
                   detections, lead_det, centerline_img, centerline_extrap,
                   raw_centerline_img, orig_w, orig_h, cam=cam, hgp_distance=hgp_dist)

    # Composite BGRA overlay onto the video frame
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay_alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_bgra[:, :, :3][:, :, ::-1].astype(np.float32)  # BGRA→RGB
    composited = frame_rgb * (1 - overlay_alpha) + overlay_rgb * overlay_alpha
    result_bgr = cv2.cvtColor(composited.astype(np.uint8), cv2.COLOR_RGB2BGR)

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


@router.get("/overlay")
async def get_overlay(filename: str, directory: str, frame: int):
    """Return a transparent PNG overlay for a processed frame."""
    ddir = _data_dir(Path(directory), filename)
    meta = _read_metadata(ddir)
    if meta is None:
        raise HTTPException(404, "Video not processed")

    batch_idx = frame // 100
    fpath = _frame_file_path(ddir, batch_idx * 100)
    if not fpath.exists():
        raise HTTPException(404, f"Batch not processed")

    raw = fpath.read_bytes()
    try:
        data = gzip.decompress(raw) if fpath.suffix == ".gz" else raw
    except gzip.BadGzipFile:
        data = raw  # legacy uncompressed remainder
    frames = json.loads(data)
    frame_idx = frame % 100
    if frame_idx >= len(frames):
        raise HTTPException(404, f"Frame {frame} not found in batch")
    frame_entry = frames[frame_idx]

    mask_payload = frame_entry.get("mask")
    if mask_payload is None:
        raise HTTPException(404, "No mask data for frame")

    orig_w = meta.get("width", _INPUT_W)
    orig_h = meta.get("height", _INPUT_H)

    # Decode mask to model-resolution class array and reconstruct full masks
    classes_region = decode_mask(mask_payload)
    classes_full = np.zeros((_INPUT_H, _INPUT_W), dtype=np.uint8)
    start_row = mask_payload["start_row"]
    classes_full[start_row:start_row + mask_payload["row_count"]] = classes_region

    da_mask = ((classes_full == DRIVEABLE) | (classes_full == LANE_ON_DRIVEABLE)).astype(np.uint8)
    lane_mask = ((classes_full == LANE_ON_DRIVEABLE) | (classes_full == LANE_NO_DRIVEABLE)).astype(np.uint8)

    cam = _load_camera_params(Path(directory))
    detections = frame_entry.get("detections", [])

    png_bytes = _render_overlay_png(da_mask, lane_mask, detections, orig_w, orig_h, cam)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


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

        if _all_steps_complete_from_dir(directory, filename):
            continue  # already complete

        _processing_videos.add(key)
        _executor.submit(_process_video_worker, filename, directory, key)
        started.append(filename)

    return {"started": started}
