"""ByteTrack multi-object tracker (pure numpy/scipy, no extra dependencies)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU between every pair of boxes in a (N,4) and b (M,4).

    Returns an (N, M) matrix.
    """
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# Single-object Kalman tracker
# ---------------------------------------------------------------------------

class KalmanTrack:
    """Constant-velocity Kalman filter for a single bounding box.

    State: [cx, cy, area, aspect_ratio, vcx, vcy, v_area]
    Measurement: [cx, cy, area, aspect_ratio]
    """

    _next_id: int = 1

    def __init__(self, bbox: np.ndarray, confidence: float) -> None:
        cx, cy, area, ar = self._bbox_to_z(bbox)

        # State vector
        self.x = np.array([cx, cy, area, ar, 0.0, 0.0, 0.0])

        # State transition (constant velocity)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0

        # Measurement matrix (observe cx, cy, area, ar)
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Process noise
        self.Q = np.diag([1.0, 1.0, 10.0, 1e-3, 0.1, 0.1, 1e-3])
        # Measurement noise
        self.R = np.diag([1.0, 1.0, 10.0, 1e-3])
        # Initial covariance — high uncertainty in velocity
        self.P = np.diag([10.0, 10.0, 100.0, 0.01, 1e4, 1e4, 1e4])

        self.id: int = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.confidence: float = confidence
        self.hits: int = 1
        self.hit_streak: int = 1
        self.time_since_update: int = 0
        self.age: int = 0

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> tuple[float, float, float, float]:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        area = float(max(w * h, 1.0))
        ar = float(w / max(h, 1e-6))
        return cx, cy, area, ar

    # ── Kalman steps ────────────────────────────────────────────────────────

    def predict(self) -> None:
        """Advance the state one step without a measurement."""
        # Clamp area to stay positive
        if self.x[2] + self.x[6] < 1.0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def update(self, bbox: np.ndarray, confidence: float) -> None:
        """Update the filter with a new measurement."""
        cx, cy, area, ar = self._bbox_to_z(bbox)
        z = np.array([cx, cy, area, ar])

        y = z - self.H @ self.x                      # Innovation
        S = self.H @ self.P @ self.H.T + self.R      # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.confidence = confidence
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

    def to_bbox(self) -> np.ndarray:
        """Return the current state as [x1, y1, x2, y2]."""
        cx, cy, area, ar = self.x[:4]
        area = max(area, 1.0)
        w = float(np.sqrt(area * max(ar, 1e-6)))
        h = float(area / max(w, 1e-6))
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """ByteTrack multi-object tracker.

    Two-stage association:
      1. Match high-confidence detections to all tracks.
      2. Match unmatched tracks to low-confidence detections.

    This lets occlusion-affected detections (low conf) keep existing tracks
    alive without promoting them to create new track IDs.

    Since the detector already filters at _CONF_THRESHOLD (0.4), we split:
      high_thresh ≥ 0.6  →  confident detections
      low_thresh  ≥ 0.4  →  border-line detections kept by the model
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.4,
        iou_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[KalmanTrack] = []
        self.frame_count: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> list[dict]:
        """Update tracker with detections from a single frame.

        Args:
            detections: list of dicts with keys x1, y1, x2, y2, confidence.

        Returns:
            list of dicts with keys track_id, x1, y1, x2, y2, confidence
            for every currently active track that passes the min_hits filter.
        """
        self.frame_count += 1

        # Split detections by confidence tier
        high = [d for d in detections if d["confidence"] >= self.high_thresh]
        low  = [d for d in detections if self.low_thresh <= d["confidence"] < self.high_thresh]

        # Predict all tracks forward one step
        for t in self.tracks:
            t.predict()

        # ── Stage 1: high-confidence dets → all tracks ──────────────────────
        matched1, unmatched_tracks1, unmatched_high = self._match(self.tracks, high)
        for ti, di in matched1:
            d = high[di]
            self.tracks[ti].update(
                np.array([d["x1"], d["y1"], d["x2"], d["y2"]]),
                d["confidence"],
            )

        # ── Stage 2: low-confidence dets → unmatched tracks only ────────────
        remaining = [self.tracks[i] for i in unmatched_tracks1]
        matched2, _, _ = self._match(remaining, low)
        for ti, di in matched2:
            d = low[di]
            remaining[ti].update(
                np.array([d["x1"], d["y1"], d["x2"], d["y2"]]),
                d["confidence"],
            )

        # ── Spawn new tracks for unmatched high-confidence detections ────────
        for di in unmatched_high:
            d = high[di]
            self.tracks.append(
                KalmanTrack(
                    np.array([d["x1"], d["y1"], d["x2"], d["y2"]]),
                    d["confidence"],
                )
            )

        # ── Prune tracks that have been lost too long ────────────────────────
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # ── Build output: only tracks that have been seen enough ─────────────
        output: list[dict] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue  # not updated this frame
            if t.hits < self.min_hits and self.frame_count > self.min_hits:
                continue  # not yet confirmed
            bbox = t.to_bbox()
            output.append({
                "track_id": t.id,
                "x1": int(round(float(bbox[0]))),
                "y1": int(round(float(bbox[1]))),
                "x2": int(round(float(bbox[2]))),
                "y2": int(round(float(bbox[3]))),
                "confidence": round(t.confidence, 3),
            })

        return output

    # ── Private helpers ─────────────────────────────────────────────────────

    def _match(
        self,
        tracks: list[KalmanTrack],
        dets: list[dict],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian matching via IoU cost.

        Returns (matched, unmatched_track_indices, unmatched_det_indices).
        """
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        track_boxes = np.array([t.to_bbox() for t in tracks])
        det_boxes   = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])

        iou   = _iou_matrix(track_boxes, det_boxes)
        cost  = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)

        matched: list[tuple[int, int]] = []
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= self.iou_thresh:
                matched.append((r, c))

        matched_t = {r for r, _ in matched}
        matched_d = {c for _, c in matched}
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_dets   = [i for i in range(len(dets))   if i not in matched_d]

        return matched, unmatched_tracks, unmatched_dets
