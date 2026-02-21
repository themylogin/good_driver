from __future__ import annotations

import atexit
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calibrate")

# Project root: backend/good_driver/api/calibrate.py → ../../../../
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "yolopv2_384x640.onnx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lazy-loaded ONNX session
_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run: uv run python download_model.py"
            )
        _session = ort.InferenceSession(str(MODEL_PATH))
        logger.info("Loaded YOLO model. Outputs:")
        for o in _session.get_outputs():
            logger.info("  %s: %s", o.name, o.shape)
    return _session


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

_INPUT_W = 640
_INPUT_H = 384
_ANCHORS = {
    8: [(12, 16), (19, 36), (40, 28)],
    16: [(36, 75), (76, 55), (72, 146)],
    32: [(142, 110), (192, 243), (459, 401)],
}
_CONF_THRESHOLD = 0.4
_IOU_THRESHOLD = 0.45


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _preprocess(img: Image.Image) -> tuple[np.ndarray, float, float]:
    orig_w, orig_h = img.size
    resized = img.resize((_INPUT_W, _INPUT_H), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0  # HWC
    arr = arr.transpose(2, 0, 1)[np.newaxis]           # NCHW
    return arr, orig_w / _INPUT_W, orig_h / _INPUT_H


def _decode_raw_predictions(outputs: list) -> list[dict]:
    """
    Decode raw YOLOPv2 prediction heads (3 scales).
    Handles multiple output formats:
      - [1, 3, grid_h, grid_w, 85]  (anchor-first)
      - [1, 255, grid_h, grid_w]    (channels = 3*85, NCHW style) ← actual PINTO format
      - [1, num_preds, 85]          (flattened)
    Returns boxes in model input space (640x384).
    """
    boxes = []
    strides = [8, 16, 32]
    for stride, pred_tensor in zip(strides, outputs[2:5]):
        anchors = _ANCHORS[stride]
        na = len(anchors)
        pred = pred_tensor[0]  # remove batch dim → shape varies

        if pred.ndim == 3 and pred.shape[0] == na * 85:
            # [255, grid_h, grid_w] → reshape to [3, 85, grid_h, grid_w] → [3, grid_h, grid_w, 85]
            nc_total, grid_h, grid_w = pred.shape
            nc = nc_total // na
            pred = pred.reshape(na, nc, grid_h, grid_w).transpose(0, 2, 3, 1)
        elif pred.ndim == 2:
            # [num_preds, 85] flattened
            grid_h = _INPUT_H // stride
            grid_w = _INPUT_W // stride
            nc = pred.shape[1]
            pred = pred.reshape(na, grid_h, grid_w, nc)
        else:
            # [3, grid_h, grid_w, 85]
            na, grid_h, grid_w, nc = pred.shape

        grid_y, grid_x = np.mgrid[:grid_h, :grid_w]

        for ai, (aw, ah) in enumerate(anchors):
            p = pred[ai]  # [grid_h, grid_w, 85]

            bx = (_sigmoid(p[..., 0]) * 2 - 0.5 + grid_x) * stride
            by = (_sigmoid(p[..., 1]) * 2 - 0.5 + grid_y) * stride
            bw = np.power(_sigmoid(p[..., 2]) * 2, 2) * aw
            bh = np.power(_sigmoid(p[..., 3]) * 2, 2) * ah

            obj_conf = _sigmoid(p[..., 4])

            # This model is a single-class vehicle detector fine-tuned from COCO weights.
            # Only one class slot is active; use max class confidence for fg/bg discrimination
            # but label everything as "Car".
            if nc > 5:
                cls_conf = _sigmoid(p[..., 5:])
                car_score = obj_conf * cls_conf.max(axis=-1)
            else:
                car_score = obj_conf

            mask = car_score > _CONF_THRESHOLD
            if not mask.any():
                continue

            for gy, gx in zip(*np.where(mask)):
                cx, cy_ = float(bx[gy, gx]), float(by[gy, gx])
                w, h = float(bw[gy, gx]), float(bh[gy, gx])
                boxes.append(
                    {
                        "x1": cx - w / 2,
                        "y1": cy_ - h / 2,
                        "x2": cx + w / 2,
                        "y2": cy_ + h / 2,
                        "confidence": float(car_score[gy, gx]),
                    }
                )
    return boxes


def _decode_postnms_detections(det: np.ndarray) -> list[dict]:
    """
    Handle post-NMS detection output: [1, N, 6] where row = [x1,y1,x2,y2,score,class_id]
    in model input space.
    """
    boxes = []
    for row in det[0]:
        x1, y1, x2, y2, score, _cls = row
        if score > _CONF_THRESHOLD:
            boxes.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(score),
                }
            )
    return boxes


def _nms(boxes: list[dict]) -> list[dict]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if _iou(best, b) < _IOU_THRESHOLD]
    return kept


def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _scale_boxes(boxes: list[dict], sx: float, sy: float) -> list[dict]:
    return [
        {
            **b,
            "x1": b["x1"] * sx,
            "y1": b["y1"] * sy,
            "x2": b["x2"] * sx,
            "y2": b["y2"] * sy,
        }
        for b in boxes
    ]


def detect_cars(image_path: Path) -> tuple[list[dict], int, int]:
    """Run YOLO on image_path. Returns (detections, orig_w, orig_h)."""
    session = _get_session()
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    tensor, sx, sy = _preprocess(img)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    # Determine output format by inspecting first detection-related output
    # Format A: ≥5 outputs → [seg, ll, pred0, pred1, pred2] (raw predictions)
    # Format B: detection output shape is [1, N, 6] (post-NMS)
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
        logger.warning("Unexpected YOLO output format. Shapes: %s", [o.shape for o in outputs])
        boxes = []

    scaled = _scale_boxes(boxes, sx, sy)
    detections = []
    for i, b in enumerate(scaled):
        x1 = max(0, int(b["x1"]))
        y1 = max(0, int(b["y1"]))
        x2 = min(orig_w, int(b["x2"]))
        y2 = min(orig_h, int(b["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "id": i,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": round(b["confidence"], 3),
        })
    return detections, orig_w, orig_h


# ---------------------------------------------------------------------------
# Sidecar JSON helpers
# ---------------------------------------------------------------------------

def _sidecar_path(directory: Path, filename: str) -> Path:
    return directory / (filename + ".json")


def _load_sidecar(directory: Path, filename: str) -> dict:
    p = _sidecar_path(directory, filename)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_sidecar(directory: Path, filename: str, data: dict) -> None:
    p = _sidecar_path(directory, filename)
    p.write_text(json.dumps(data, indent=2))


def _image_entry(directory: Path, filename: str) -> dict:
    sidecar = _load_sidecar(directory, filename)
    return {
        "filename": filename,
        "image_url": f"/api/calibrate/image?path={directory.name}/{filename}",
        "image_width": sidecar.get("image_width"),
        "image_height": sidecar.get("image_height"),
        "detections": sidecar.get("detections"),
        "selected_detection_id": sidecar.get("selected_detection_id"),
        "car_width_m": sidecar.get("car_width_m"),
        "distance_m": sidecar.get("distance_m"),
    }


# ---------------------------------------------------------------------------
# Camera math solver
# ---------------------------------------------------------------------------

def solve_camera_params(
    measurements: list[dict],
    image_width: int,
    image_height: int,
) -> dict:
    """
    Fit [fx, pitch, h_cam] from car bbox measurements.

    Each measurement must have: pw, y_base, car_width_m, distance_m, filename.
    """
    cy = image_height / 2.0

    def residuals(params: np.ndarray) -> np.ndarray:
        fx, pitch, h_cam = params
        res = []
        for m in measurements:
            pw = m["pw"]
            y_base = m["y_base"]
            d = m["distance_m"]
            cw = m["car_width_m"]
            # Width equation
            res.append(fx * cw / d - pw)
            # Base equation
            angle = math.atan2(h_cam, d) - pitch
            res.append(cy + fx * math.tan(angle) - y_base)
        return np.array(res, dtype=float)

    m0 = measurements[0]
    fx0 = max(100.0, m0["pw"] * m0["distance_m"] / m0["car_width_m"])

    result = least_squares(
        residuals,
        x0=[fx0, -0.05, 1.2],
        bounds=([100, -0.7, 0.1], [10000, 0.7, 5.0]),
        method="trf",
    )
    fx, pitch, h_solved = result.x

    fov_deg = 2.0 * math.degrees(math.atan(image_width / 2.0 / fx))
    pitch_deg = math.degrees(pitch)

    per_image = []
    for m in measurements:
        d_est = fx * m["car_width_m"] / m["pw"] if m["pw"] > 0 else None
        per_image.append(
            {
                "filename": m["filename"],
                "estimated_distance_m": round(d_est, 1) if d_est else None,
            }
        )

    return {
        "fov_degrees": round(fov_deg, 1),
        "pitch_degrees": round(pitch_deg, 1),
        "camera_height_m": round(h_solved, 2),
        "fx": round(fx, 1),
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/open-directory")
async def open_directory():
    """Open the debug data directory and return its image list."""
    if not DATA_DIR.exists():
        raise HTTPException(404, f"Data directory not found: {DATA_DIR}")

    images = sorted(
        f.name
        for f in DATA_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    return {
        "directory": str(DATA_DIR),
        "images": [_image_entry(DATA_DIR, fn) for fn in images],
    }


@router.get("/image")
async def serve_image(path: str):
    """Serve an image from the data directory. path = e.g. 'data/1.jpg'"""
    # Sanitize: only allow files inside PROJECT_ROOT
    resolved = (PROJECT_ROOT / path).resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT)):
        raise HTTPException(403, "Access denied")
    if not resolved.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(resolved))


class DetectRequest(BaseModel):
    directory: str
    filename: str


@router.post("/detect")
async def detect(req: DetectRequest):
    """Run YOLO on an image (or return cached detections from sidecar JSON)."""
    directory = Path(req.directory)
    image_path = directory / req.filename

    if not image_path.exists():
        raise HTTPException(404, f"Image not found: {image_path}")

    sidecar = _load_sidecar(directory, req.filename)

    # Return cached detections if available
    if sidecar.get("detections") is not None:
        return _image_entry(directory, req.filename)

    # Run YOLO
    detections, orig_w, orig_h = detect_cars(image_path)

    # Merge into sidecar
    sidecar["image_width"] = orig_w
    sidecar["image_height"] = orig_h
    sidecar["detections"] = detections
    _save_sidecar(directory, req.filename, sidecar)

    return _image_entry(directory, req.filename)


class AnnotationRequest(BaseModel):
    directory: str
    filename: str
    selected_detection_id: Optional[int] = None
    car_width_m: Optional[float] = None
    distance_m: Optional[float] = None


@router.put("/annotation")
async def save_annotation(req: AnnotationRequest):
    """Persist user selections for an image."""
    directory = Path(req.directory)
    sidecar = _load_sidecar(directory, req.filename)

    if req.selected_detection_id is not None:
        sidecar["selected_detection_id"] = req.selected_detection_id
    if req.car_width_m is not None:
        sidecar["car_width_m"] = req.car_width_m
    if req.distance_m is not None:
        sidecar["distance_m"] = req.distance_m

    _save_sidecar(directory, req.filename, sidecar)
    return {"ok": True}


def _collect_measurements(directory: Path) -> list[dict]:
    """Read all sidecar files in directory and return fully-annotated measurements."""
    measurements = []
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        sidecar = _load_sidecar(directory, f.name)
        sel_id = sidecar.get("selected_detection_id")
        car_width = sidecar.get("car_width_m")
        distance = sidecar.get("distance_m")
        detections = sidecar.get("detections")

        if sel_id is None or car_width is None or distance is None or not detections:
            continue

        det = next((d for d in detections if d["id"] == sel_id), None)
        if det is None:
            continue

        pw = det["x2"] - det["x1"]
        y_base = det["y2"]

        if pw <= 0:
            continue

        measurements.append(
            {
                "filename": f.name,
                "pw": pw,
                "y_base": y_base,
                "car_width_m": float(car_width),
                "distance_m": float(distance),
                "image_width": sidecar.get("image_width", 1920),
                "image_height": sidecar.get("image_height", 1080),
            }
        )
    return measurements


class SolveRequest(BaseModel):
    directory: str


@router.post("/solve")
async def solve(req: SolveRequest):
    """Read all annotated images in directory and solve for FOV + pitch + height."""
    measurements = _collect_measurements(Path(req.directory))

    if len(measurements) < 2:
        raise HTTPException(
            400,
            f"Need at least 2 complete measurements, got {len(measurements)}",
        )

    image_width = measurements[0]["image_width"]
    image_height = measurements[0]["image_height"]
    return solve_camera_params(measurements, image_width, image_height)


@router.get("/params")
async def get_camera_params():
    """Return solved camera parameters using the data directory's calibration images.

    Returns 404 if fewer than 2 images are fully annotated.
    Includes image_width and image_height so callers can scale fx to other resolutions.
    """
    measurements = _collect_measurements(DATA_DIR)

    if len(measurements) < 2:
        raise HTTPException(
            404,
            f"Need at least 2 calibrated images, have {len(measurements)}",
        )

    image_width = measurements[0]["image_width"]
    image_height = measurements[0]["image_height"]
    result = solve_camera_params(measurements, image_width, image_height)
    result["image_width"] = image_width
    result["image_height"] = image_height
    return result
