import { useCallback, useEffect, useRef, useState } from "react";
import type { CameraParams } from "../App";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Detection {
  track_id: number;
  x1: number; y1: number; x2: number; y2: number;
  confidence: number;
}

interface FrameEntry {
  frame: number;
  detections: Detection[];
  lane_lines: number[][][];  // [line][[x,y],...]
}

type VideoMeta = { total_frames: number; processed_frames: number; fps?: number; processing?: boolean } | null;

/** A detected vehicle assigned to a lane. */
interface CarInLane {
  track_id: number;
  distance: number;  // metres forward
}

/**
 * lanes[0]  = ego lane
 * lanes[1]  = one lane to the left
 * lanes[-1] = one lane to the right
 * etc.
 */
type LaneMap = Map<number, CarInLane[]>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_FPS = 30;

const BOX_COLORS = [
  "#E53935", "#43A047", "#1E88E5", "#FB8C00",
  "#8E24AA", "#00ACC1", "#F4511E", "#039BE5",
];

const BEV_MAX_DIST = 100;  // meters forward
const BEV_HALF_WIDTH = 8;  // meters each side

// Schematic (equal-width) lanes BEV
const SCHEMA_MIN_LANE = -2;  // rightmost lane shown  (negative = right of ego)
const SCHEMA_MAX_LANE = 2;   // leftmost lane shown   (positive = left of ego)
const SCHEMA_N_LANES  = SCHEMA_MAX_LANE - SCHEMA_MIN_LANE + 1;  // 5

const btnStyle: React.CSSProperties = {
  padding: "0.3rem 0.7rem",
  border: "1px solid #ccc",
  borderRadius: "4px",
  background: "#f5f5f5",
  cursor: "pointer",
  fontSize: "0.85rem",
  fontFamily: "system-ui",
  height: "30px",
  lineHeight: "1",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(t: number): string {
  if (!isFinite(t) || isNaN(t)) return "0:00.00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const cs = Math.floor((t % 1) * 100);
  return `${m}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

/**
 * Estimate distance to a detected object using solved camera parameters.
 * yBase: bottom edge of bounding box in VIDEO pixel coords.
 */
function estimateDistance(
  yBase: number,
  videoWidth: number,
  videoHeight: number,
  params: CameraParams,
): number | null {
  // Scale fx from calibration image resolution to video resolution
  const scale = videoWidth / params.image_width;
  const fx = params.fx * scale;
  const pitchRad = params.pitch_degrees * (Math.PI / 180);
  const cy = videoHeight / 2;
  const beta = Math.atan((yBase - cy) / fx);
  const d = params.camera_height_m / Math.tan(beta + pitchRad);
  return d > 0 && isFinite(d) && d < 500 ? Math.round(d * 10) / 10 : null;
}

/**
 * Back-project an image pixel to world ground-plane coordinates.
 * Returns { X: lateral metres (right+), Z: forward metres } or null if above horizon.
 */
function imageToWorld(
  u: number, v: number,
  videoW: number, videoH: number,
  params: CameraParams,
): { X: number; Z: number } | null {
  if (!params.fx || !params.image_width || !videoW || !videoH) return null;
  const fxScaled = params.fx * (videoW / params.image_width);
  const cx = videoW / 2;
  const cy = videoH / 2;
  const pitchRad = params.pitch_degrees * (Math.PI / 180);
  const beta = Math.atan((v - cy) / fxScaled);
  const angle = beta + pitchRad;
  if (Math.tan(angle) <= 0) return null;  // point above or on horizon
  const Z = params.camera_height_m / Math.tan(angle);
  const X = (u - cx) * Z / fxScaled;
  return { X, Z };
}

/** Compute letterbox transform for object-fit:contain video inside a canvas. */
function letterboxTransform(
  canvasW: number, canvasH: number,
  videoW: number, videoH: number,
) {
  if (!videoW || !videoH) return null;
  const scale = Math.min(canvasW / videoW, canvasH / videoH);
  return {
    scale,
    offsetX: (canvasW - videoW * scale) / 2,
    offsetY: (canvasH - videoH * scale) / 2,
  };
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------

function drawOverlay(
  canvas: HTMLCanvasElement,
  videoEl: HTMLVideoElement,
  frame: FrameEntry | null,
  params: CameraParams,
) {
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!frame) return;

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  const tx = letterboxTransform(canvas.width, canvas.height, vw, vh);
  if (!tx) return;
  const { scale, offsetX, offsetY } = tx;

  // ── Lane lines ──
  ctx.strokeStyle = "#FFD700";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  for (const line of frame.lane_lines) {
    if (line.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(line[0][0] * scale + offsetX, line[0][1] * scale + offsetY);
    for (let i = 1; i < line.length; i++) {
      ctx.lineTo(line[i][0] * scale + offsetX, line[i][1] * scale + offsetY);
    }
    ctx.stroke();
  }

  // ── Bounding boxes + labels ──
  for (const det of frame.detections) {
    const color = BOX_COLORS[det.track_id % BOX_COLORS.length];
    const cx1 = det.x1 * scale + offsetX;
    const cy1 = det.y1 * scale + offsetY;
    const bw  = (det.x2 - det.x1) * scale;
    const bh  = (det.y2 - det.y1) * scale;

    // Box border
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(cx1, cy1, bw, bh);

    // Distance label (inside top-left of box)
    const dist = estimateDistance(det.y2, vw, vh, params);
    const label = dist !== null ? `${dist} m` : `${Math.round(det.confidence * 100)}%`;
    const fontSize = Math.max(11, Math.min(16, bw / 6));
    ctx.font = `bold ${fontSize}px system-ui`;
    const tw = ctx.measureText(label).width;
    const lh = fontSize + 4;
    const lx = cx1 + 2;
    const ly = cy1 + 2;

    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, tw + 8, lh);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, lx + 4, ly + fontSize - 1);
  }
}

// ---------------------------------------------------------------------------
// Lane fitting helpers
// ---------------------------------------------------------------------------


/** Least-squares linear fit: X = a·Z + b */
function fitLinear(pts: { X: number; Z: number }[]): { a: number; b: number } | null {
  const n = pts.length;
  if (n < 2) return null;
  let sumZ = 0, sumX = 0, sumZZ = 0, sumXZ = 0;
  for (const { X, Z } of pts) {
    sumZ += Z; sumX += X; sumZZ += Z * Z; sumXZ += X * Z;
  }
  const denom = n * sumZZ - sumZ * sumZ;
  if (Math.abs(denom) < 1e-9) return null;
  const a = (n * sumXZ - sumX * sumZ) / denom;
  const b = (sumX - a * sumZ) / n;
  return { a, b };
}

/**
 * Project + fit all lane lines in a frame to parallel world-space lines.
 * Returns deduped array sorted left-to-right.
 */
function computeFittedLines(
  frame: FrameEntry,
  videoW: number,
  videoH: number,
  params: CameraParams,
): { a: number; b: number }[] {
  const linePts: { X: number; Z: number }[][] = [];
  for (const line of frame.lane_lines) {
    const pts = line
      .map(([u, v]) => imageToWorld(u, v, videoW, videoH, params))
      .filter(Boolean) as { X: number; Z: number }[];
    if (pts.length >= 2) linePts.push(pts);
  }
  if (linePts.length === 0) return [];

  const indivFits = linePts.map(fitLinear).filter(Boolean) as { a: number; b: number }[];
  const sortedSlopes = indivFits.map(f => f.a).sort((p, q) => p - q);
  const sharedA = sortedSlopes.length > 0
    ? sortedSlopes[Math.floor(sortedSlopes.length / 2)]
    : 0;

  const fitted = linePts.map(pts => {
    const b = pts.reduce((s, { X, Z }) => s + X - sharedA * Z, 0) / pts.length;
    return { a: sharedA, b };
  });

  const REF_Z = 10;
  fitted.sort((p, q) => (p.a * REF_Z + p.b) - (q.a * REF_Z + q.b));

  const MIN_LANE_GAP = 1.0;
  return fitted.filter((line, i) => {
    if (i === 0) return true;
    const prevX = fitted[i - 1].a * REF_Z + fitted[i - 1].b;
    const thisX = line.a * REF_Z + line.b;
    return thisX - prevX >= MIN_LANE_GAP;
  });
}

/**
 * Assign each detected vehicle to a lane index relative to the ego vehicle.
 *   0  = ego's lane
 *  +1  = one lane to the left
 *  -1  = one lane to the right
 */
function computeLanes(
  frame: FrameEntry,
  deduped: { a: number; b: number }[],
  videoW: number,
  videoH: number,
  params: CameraParams,
): LaneMap {
  const lanes: LaneMap = new Map();
  if (!videoW || !videoH) return lanes;

  // Find which interval the ego vehicle (X=0) falls in at a small near depth
  const EGO_Z = 2;
  const lineXsAtEgo = deduped.map(({ a, b }) => a * EGO_Z + b).sort((p, q) => p - q);

  function intervalOf(X: number, sortedXs: number[]): number {
    // Returns 0 for X < sortedXs[0], k for sortedXs[k-1] <= X < sortedXs[k], n for X >= sortedXs[n-1]
    for (let i = 0; i < sortedXs.length; i++) {
      if (X < sortedXs[i]) return i;
    }
    return sortedXs.length;
  }

  const egoInterval = intervalOf(0, lineXsAtEgo);

  for (const det of frame.detections) {
    const u = (det.x1 + det.x2) / 2;
    const v = det.y2;
    const world = imageToWorld(u, v, videoW, videoH, params);
    if (!world) continue;
    const { X: carX, Z: carZ } = world;

    const lineXsAtCar = deduped.map(({ a, b }) => a * carZ + b).sort((p, q) => p - q);
    const carInterval = intervalOf(carX, lineXsAtCar);

    // egoInterval - carInterval:  positive = car is to the LEFT  (lane +1, +2…)
    //                             negative = car is to the RIGHT (lane -1, -2…)
    const laneIdx = egoInterval - carInterval;

    const entry: CarInLane = { track_id: det.track_id, distance: Math.round(carZ * 10) / 10 };
    if (!lanes.has(laneIdx)) lanes.set(laneIdx, []);
    lanes.get(laneIdx)!.push(entry);
  }
  return lanes;
}

// ---------------------------------------------------------------------------
// Bird's Eye View drawing
// ---------------------------------------------------------------------------

function drawBEV(
  bevCanvas: HTMLCanvasElement,
  frame: FrameEntry | null,
  params: CameraParams,
  videoW: number,
  videoH: number,
) {
  const ctx = bevCanvas.getContext("2d")!;
  const W = bevCanvas.width;
  const H = bevCanvas.height;
  if (!W || !H) return;

  // Pixels per metre – fit BEV_MAX_DIST vertically and BEV_HALF_WIDTH*2 horizontally
  const scaleZ = H / BEV_MAX_DIST;
  const scaleX = (W / 2) / BEV_HALF_WIDTH;
  const ppm = Math.min(scaleZ, scaleX);  // pixels per metre

  const cxBEV = W / 2;

  function worldToBEV(X: number, Z: number): [number, number] {
    return [cxBEV + X * ppm, H - Z * ppm];
  }

  // Background
  ctx.fillStyle = "#464646";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.lineWidth = 1;
  ctx.font = "11px system-ui";
  ctx.textAlign = "left";

  // Centre (forward) line
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.beginPath();
  ctx.moveTo(cxBEV, H);
  ctx.lineTo(cxBEV, 0);
  ctx.stroke();

  // Distance rings
  for (let z = 5; z <= BEV_MAX_DIST; z += 5) {
    const [, gy] = worldToBEV(0, z);
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(W, gy);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText(`${z}m`, cxBEV + 4, gy - 3);
  }

  if (!frame || !videoW || !videoH) return;

  // ── Lane lines ──
  ctx.strokeStyle = "#FFD700";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  for (const line of frame.lane_lines) {
    const pts = line
      .map(([u, v]) => imageToWorld(u, v, videoW, videoH, params))
      .filter(Boolean) as { X: number; Z: number }[];
    if (pts.length < 2) continue;
    ctx.beginPath();
    const [bx0, by0] = worldToBEV(pts[0].X, pts[0].Z);
    ctx.moveTo(bx0, by0);
    for (let i = 1; i < pts.length; i++) {
      const [bx, by] = worldToBEV(pts[i].X, pts[i].Z);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();
  }

  // ── Vehicles ──
  for (const det of frame.detections) {
    const u = (det.x1 + det.x2) / 2;  // horizontal centre
    const v = det.y2;                  // bottom edge = ground contact
    const world = imageToWorld(u, v, videoW, videoH, params);
    if (!world) continue;
    const [bx, by] = worldToBEV(world.X, world.Z);
    if (bx < -10 || bx > W + 10 || by < -10 || by > H + 10) continue;
    ctx.beginPath();
    ctx.arc(bx, by, 8, 0, Math.PI * 2);
    ctx.fillStyle = BOX_COLORS[det.track_id % BOX_COLORS.length];
    ctx.fill();
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Schematic lanes BEV (equal-width parallel lanes, cars snapped to lane centre)
// ---------------------------------------------------------------------------

function drawLanesBEV(
  bevCanvas: HTMLCanvasElement,
  frame: FrameEntry | null,
  params: CameraParams,
  videoW: number,
  videoH: number,
) {
  const ctx = bevCanvas.getContext("2d")!;
  const W = bevCanvas.width;
  const H = bevCanvas.height;
  if (!W || !H) return;

  const laneW = W / SCHEMA_N_LANES;

  // Map lane index → canvas X centre.
  // Positive lane index = left of ego → left side of canvas.
  // col = SCHEMA_MAX_LANE - laneIdx  (flips the axis)
  function laneCenterX(laneIdx: number): number {
    const col = Math.max(0, Math.min(SCHEMA_N_LANES - 1, SCHEMA_MAX_LANE - laneIdx));
    return (col + 0.5) * laneW;
  }
  function distToY(dist: number): number {
    return H - dist * H / BEV_MAX_DIST;
  }

  // ── Background ──
  ctx.fillStyle = "#464646";
  ctx.fillRect(0, 0, W, H);

  // ── Ego-lane highlight ──  (col for laneIdx=0 is SCHEMA_MAX_LANE)
  ctx.fillStyle = "rgba(255,255,255,0.06)";
  ctx.fillRect(SCHEMA_MAX_LANE * laneW, 0, laneW, H);

  // ── Distance grid ──
  ctx.lineWidth = 1;
  ctx.font = "11px system-ui";
  ctx.textAlign = "left";
  const egoLabelX = laneCenterX(0) + 4;
  for (let z = 5; z <= BEV_MAX_DIST; z += 5) {
    const gy = distToY(z);
    ctx.strokeStyle = "rgba(255,255,255,0.25)";
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(W, gy);
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText(`${z}m`, egoLabelX, gy - 3);
  }

  // ── Lane boundary lines ──
  ctx.strokeStyle = "#FFD700";
  ctx.lineWidth = 2;
  for (let k = 0; k <= SCHEMA_N_LANES; k++) {
    const x = k * laneW;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  // ── Ego vehicle marker ──
  ctx.beginPath();
  ctx.arc(laneCenterX(0), H - 12, 6, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.fill();

  if (!frame || !videoW || !videoH) return;

  // ── Compute lane assignments ──
  const deduped = computeFittedLines(frame, videoW, videoH, params);
  const lanes    = computeLanes(frame, deduped, videoW, videoH, params);

  // ── Draw vehicles ──
  for (const [laneIdx, cars] of lanes) {
    const x = laneCenterX(laneIdx);
    for (const car of cars) {
      const y = distToY(car.distance);
      if (y < -10 || y > H + 10) continue;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = BOX_COLORS[car.track_id % BOX_COLORS.length];
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

// ---------------------------------------------------------------------------
// Subtitle component
// ---------------------------------------------------------------------------

interface ProgressSnapshot {
  frames: number;
  time: number;
}

function formatEta(seconds: number): string {
  const total = Math.ceil(seconds);
  if (total < 60) return `${total}s`;
  const m = Math.floor(total / 60);
  const s = total % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function VideoSubtitle({ filename, meta, progressRef }: { filename: string; meta: VideoMeta; progressRef: React.RefObject<Record<string, ProgressSnapshot>> }) {
  if (!meta || meta.total_frames === 0) {
    return <div style={{ color: "#aaa", fontSize: "0.78rem" }}>Not processed</div>;
  }
  if (meta.processed_frames >= meta.total_frames) {
    return <div style={{ color: "#4a0", fontSize: "0.78rem" }}>Processing complete</div>;
  }
  const pct = Math.round((meta.processed_frames / meta.total_frames) * 100);

  // Compute ETA from first observation of progress
  let etaStr = "";
  const snap = progressRef.current?.[filename];
  if (snap && meta.processed_frames > snap.frames) {
    const elapsed = (Date.now() - snap.time) / 1000;
    const done = meta.processed_frames - snap.frames;
    const remaining = meta.total_frames - meta.processed_frames;
    const rate = done / elapsed;
    if (rate > 0) {
      etaStr = `ETA ${formatEta(remaining / rate)}`;
    }
  }

  return (
    <div style={{ color: "#888", fontSize: "0.78rem" }}>
      {meta.processed_frames}/{meta.total_frames} frames ({pct}%){etaStr && `, ${etaStr}`}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface UploadFootageProps {
  directory: string;
  cameraParams: CameraParams;
}

interface VideoEntry {
  filename: string;
  video_url: string;
}

export default function UploadFootage({ directory, cameraParams }: UploadFootageProps) {
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [activeVideo, setActiveVideo] = useState<VideoEntry | null>(null);
  const [metas, setMetas] = useState<Record<string, VideoMeta>>({});
  const [processingStarted, setProcessingStarted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [fps, setFps] = useState(DEFAULT_FPS);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bevCanvasRef = useRef<HTMLCanvasElement>(null);
  const lanesBevCanvasRef = useRef<HTMLCanvasElement>(null);
  const isScrubbing = useRef(false);
  // Cache batch data: batchIndex → FrameEntry[]
  const batchCache = useRef<Record<number, FrameEntry[]>>({});
  const frameDataRef = useRef<FrameEntry | null>(null);
  // ETA tracking: filename → first observed {frames, time} for current processing run
  const progressSnapshots = useRef<Record<string, ProgressSnapshot>>({});

  // ── Fetch metadata for all videos ──────────────────────────────────────
  const fetchAllMeta = useCallback(async (videoList: VideoEntry[]) => {
    const entries = await Promise.all(
      videoList.map(async (v) => {
        const r = await fetch(`/api/footage/metadata?filename=${encodeURIComponent(v.filename)}&directory=${encodeURIComponent(directory)}`);
        const data = r.ok ? await r.json() : null;
        return [v.filename, data] as [string, VideoMeta];
      }),
    );
    const metaMap = Object.fromEntries(entries);
    setMetas(metaMap);
    // Detect active processing from backend and seed ETA snapshots
    const now = Date.now();
    let anyProcessing = false;
    for (const [fname, m] of Object.entries(metaMap) as [string, VideoMeta][]) {
      if (m?.processing) {
        anyProcessing = true;
        if (!progressSnapshots.current[fname]) {
          progressSnapshots.current[fname] = { frames: m.processed_frames, time: now };
        }
      }
    }
    if (anyProcessing) setProcessingStarted(true);
  }, [directory]);

  // ── Load video list on mount ────────────────────────────────────────────
  useEffect(() => {
    fetch(`/api/footage/list?directory=${encodeURIComponent(directory)}`)
      .then((r) => r.json())
      .then((data) => {
        const vids: VideoEntry[] = data.videos ?? [];
        setVideos(vids);
        if (vids.length > 0) setActiveVideo(vids[0]);
        fetchAllMeta(vids);
      })
      .catch((e) => setErrorMessage(String(e)));
  }, [directory, fetchAllMeta]);

  // ── Set fps from metadata when a video is selected ─────────────────────
  useEffect(() => {
    if (!activeVideo) return;
    const meta = metas[activeVideo.filename];
    if (meta?.fps) setFps(meta.fps);
    // Clear stale frame data when switching video
    batchCache.current = {};
    frameDataRef.current = null;
  }, [activeVideo, metas]);

  // ── Poll metadata while processing is active ────────────────────────────
  const allComplete = videos.length > 0 && videos.every((v) => {
    const m = metas[v.filename];
    return m !== null && m !== undefined && m.total_frames > 0 && m.processed_frames >= m.total_frames;
  });
  // Clear the flag only when everything finishes
  useEffect(() => {
    if (allComplete) setProcessingStarted(false);
  }, [allComplete]);
  useEffect(() => {
    if (!processingStarted || videos.length === 0) return;
    const id = setInterval(() => fetchAllMeta(videos), 1000);
    return () => clearInterval(id);
  }, [processingStarted, videos, fetchAllMeta]);

  // ── Canvas drawing ──────────────────────────────────────────────────────
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const bevCanvas = bevCanvasRef.current;
    const lanesBevCanvas = lanesBevCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    // Sync canvas resolution to its CSS size
    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }
    drawOverlay(canvas, video, frameDataRef.current, cameraParams);
    // Raw BEV
    if (bevCanvas) {
      if (bevCanvas.width !== bevCanvas.clientWidth || bevCanvas.height !== bevCanvas.clientHeight) {
        bevCanvas.width = bevCanvas.clientWidth;
        bevCanvas.height = bevCanvas.clientHeight;
      }
      drawBEV(bevCanvas, frameDataRef.current, cameraParams, video.videoWidth, video.videoHeight);
    }
    // Lanes BEV
    if (lanesBevCanvas) {
      if (lanesBevCanvas.width !== lanesBevCanvas.clientWidth || lanesBevCanvas.height !== lanesBevCanvas.clientHeight) {
        lanesBevCanvas.width = lanesBevCanvas.clientWidth;
        lanesBevCanvas.height = lanesBevCanvas.clientHeight;
      }
      drawLanesBEV(lanesBevCanvas, frameDataRef.current, cameraParams, video.videoWidth, video.videoHeight);
    }
  }, [cameraParams]);

  // Redraw when any canvas is resized
  useEffect(() => {
    const canvas = canvasRef.current;
    const bevCanvas = bevCanvasRef.current;
    const lanesBevCanvas = lanesBevCanvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);
    if (bevCanvas) ro.observe(bevCanvas);
    if (lanesBevCanvas) ro.observe(lanesBevCanvas);
    return () => ro.disconnect();
  }, [redraw]);

  // ── Load frame data for the given time and redraw ───────────────────────
  const loadFrameAndDraw = useCallback(async (time: number) => {
    if (!activeVideo) return;
    const filename = activeVideo.filename;

    const frameN = Math.floor(time * fps);
    const batchIdx = Math.floor(frameN / 100);

    // Cache hit — instant
    if (batchCache.current[batchIdx]) {
      frameDataRef.current = batchCache.current[batchIdx][frameN % 100] ?? null;
      redraw();
      return;
    }

    // Fetch batch
    try {
      const res = await fetch(
        `/api/footage/frames?filename=${encodeURIComponent(filename)}&batch=${batchIdx}&directory=${encodeURIComponent(directory)}`,
      );
      if (!res.ok) {
        // Batch not yet processed — clear overlay
        frameDataRef.current = null;
        redraw();
        return;
      }
      const batch: FrameEntry[] = await res.json();
      batchCache.current[batchIdx] = batch;
      frameDataRef.current = batch[frameN % 100] ?? null;
    } catch {
      frameDataRef.current = null;
    }
    redraw();
  }, [activeVideo, fps, redraw]);

  // ── Keyboard shortcuts ──────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const vid = videoRef.current;
      if (!vid || !activeVideo) return;
      if (e.key === " ") {
        e.preventDefault();
        vid.paused ? vid.play() : vid.pause();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        vid.currentTime = Math.max(0, vid.currentTime - 1 / fps);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        vid.currentTime = Math.min(vid.duration || 0, vid.currentTime + 1 / fps);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [fps, activeVideo]);

  const stepFrame = useCallback(
    (direction: 1 | -1) => {
      const vid = videoRef.current;
      if (!vid) return;
      vid.pause();
      vid.currentTime = Math.max(0, Math.min(vid.duration || 0, vid.currentTime + direction / fps));
    },
    [fps],
  );

  const handlePlayPause = useCallback(() => {
    const vid = videoRef.current;
    if (!vid) return;
    vid.paused ? vid.play() : vid.pause();
  }, []);

  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const vid = videoRef.current;
    if (!vid) return;
    const t = parseFloat(e.target.value);
    vid.currentTime = t;
    setCurrentTime(t);
  }, []);

  // ── "Start processing" ──────────────────────────────────────────────────
  const handleStartProcessing = async () => {
    setProcessingStarted(true);
    // Record current progress as baseline for ETA calculation
    const now = Date.now();
    for (const v of videos) {
      const m = metas[v.filename];
      if (m && m.total_frames > 0 && m.processed_frames < m.total_frames) {
        progressSnapshots.current[v.filename] = { frames: m.processed_frames, time: now };
      }
    }
    try {
      const res = await fetch(`/api/footage/start-processing?directory=${encodeURIComponent(directory)}`, { method: "POST" });
      const data = await res.json();
      const count = (data.started as string[])?.length ?? 0;
      if (count === 0) setProcessingStarted(false);
      await fetchAllMeta(videos);
    } catch (e) {
      setProcessingStarted(false);
      setErrorMessage(String(e));
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      {/* ── LEFT SIDEBAR ── */}
      <div
        style={{
          width: "300px",
          flexShrink: 0,
          borderRight: "1px solid #ddd",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem" }}>
          {videos.length === 0 && (
            <div style={{ color: "#999", fontSize: "0.85rem", padding: "0.5rem" }}>
              No videos found in directory
            </div>
          )}
          {videos.map((v) => (
            <div
              key={v.filename}
              onClick={() => setActiveVideo(v)}
              style={{
                padding: "0.4rem 0.6rem",
                marginBottom: "2px",
                cursor: "pointer",
                borderRadius: "4px",
                background: activeVideo?.filename === v.filename ? "#e8f0fe" : "transparent",
                fontSize: "0.875rem",
              }}
            >
              <div
                style={{
                  fontWeight: activeVideo?.filename === v.filename ? 600 : 400,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  color: "#000",
                }}
              >
                {v.filename}
              </div>
              <VideoSubtitle filename={v.filename} meta={metas[v.filename] ?? null} progressRef={progressSnapshots} />
            </div>
          ))}
        </div>

        {/* Start processing button */}
        <div style={{ borderTop: "1px solid #ddd", padding: "0.75rem", flexShrink: 0 }}>
          <button
            onClick={handleStartProcessing}
            disabled={allComplete || processingStarted || videos.length === 0}
            style={{
              width: "100%",
              padding: "0.5rem",
              border: "1px solid #ccc",
              borderRadius: "4px",
              background: allComplete || processingStarted ? "#f0f0f0" : "#0066cc",
              color: allComplete || processingStarted ? "#999" : "#fff",
              cursor: allComplete || processingStarted ? "default" : "pointer",
              fontSize: "0.9rem",
              fontFamily: "system-ui",
            }}
          >
            {processingStarted
              ? "Processing…"
              : allComplete
              ? "All complete"
              : "Start processing"}
          </button>
        </div>
      </div>

      {/* ── RIGHT PANEL ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
        {activeVideo ? (
          <>
            {/* Video + canvas overlay + BEV */}
            <div style={{ flex: 1, background: "#1a1a1a", overflow: "hidden", display: "flex" }}>
              {/* Video + detection overlay */}
              <div style={{ flex: 1, position: "relative", minWidth: 0 }}>
                <video
                  key={activeVideo.filename}
                  ref={videoRef}
                  src={activeVideo.video_url}
                  style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onTimeUpdate={() => {
                    const vid = videoRef.current;
                    if (!vid) return;
                    if (!isScrubbing.current) setCurrentTime(vid.currentTime);
                    loadFrameAndDraw(vid.currentTime);
                  }}
                  onSeeked={() => {
                    const vid = videoRef.current;
                    if (vid) loadFrameAndDraw(vid.currentTime);
                  }}
                  onLoadedMetadata={() => {
                    const vid = videoRef.current;
                    if (vid) {
                      setDuration(vid.duration);
                      setCurrentTime(0);
                      setIsPlaying(false);
                      loadFrameAndDraw(0);
                    }
                  }}
                />
                <canvas
                  ref={canvasRef}
                  style={{
                    position: "absolute",
                    top: 0, left: 0,
                    width: "100%", height: "100%",
                    pointerEvents: "none",
                  }}
                />
              </div>
              {/* Bird's Eye View (schematic lanes) */}
              <div style={{ width: "220px", flexShrink: 0, borderLeft: "3px solid #fff", display: "flex", flexDirection: "column" }}>
                <div style={{ background: "#2a2a2a", color: "#fff", textAlign: "center", fontSize: "0.75rem", fontFamily: "system-ui", padding: "3px 0", flexShrink: 0, letterSpacing: "0.05em" }}>
                  processed
                </div>
                <canvas
                  ref={lanesBevCanvasRef}
                  style={{ width: "100%", flex: 1, display: "block" }}
                />
              </div>
              {/* Bird's Eye View (raw) */}
              <div style={{ width: "220px", flexShrink: 0, borderLeft: "3px solid #fff", display: "flex", flexDirection: "column" }}>
                <div style={{ background: "#2a2a2a", color: "#fff", textAlign: "center", fontSize: "0.75rem", fontFamily: "system-ui", padding: "3px 0", flexShrink: 0, letterSpacing: "0.05em" }}>
                  original
                </div>
                <canvas
                  ref={bevCanvasRef}
                  style={{ width: "100%", flex: 1, display: "block" }}
                />
              </div>
            </div>

            {/* Controls */}
            <div
              style={{
                borderTop: "1px solid #ddd",
                padding: "0.5rem 1rem",
                background: "#fff",
                flexShrink: 0,
                display: "flex",
                flexDirection: "column",
                gap: "0.4rem",
              }}
            >
              <input
                type="range"
                min={0}
                max={duration || 0}
                step="any"
                value={currentTime}
                onMouseDown={() => { isScrubbing.current = true; }}
                onChange={handleSliderChange}
                onMouseUp={() => { isScrubbing.current = false; }}
                style={{ width: "100%", cursor: "pointer" }}
              />
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <button onClick={() => stepFrame(-1)} style={btnStyle}>◀ Prev</button>
                <button onClick={handlePlayPause} style={{ ...btnStyle, minWidth: "76px" }}>
                  {isPlaying ? "⏸ Pause" : "▶ Play"}
                </button>
                <button onClick={() => stepFrame(1)} style={btnStyle}>Next ▶</button>
                <span
                  style={{
                    marginLeft: "0.5rem",
                    fontSize: "0.85rem",
                    color: "#555",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
                <label
                  style={{
                    marginLeft: "auto",
                    fontSize: "0.8rem",
                    color: "#777",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.3rem",
                  }}
                >
                  FPS:
                  <input
                    type="number"
                    value={fps}
                    min={1}
                    max={120}
                    onChange={(e) => setFps(Math.max(1, Number(e.target.value) || DEFAULT_FPS))}
                    style={{ width: "48px" }}
                  />
                </label>
              </div>
            </div>
          </>
        ) : (
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#999",
            }}
          >
            {videos.length === 0 ? "No videos in directory" : "Select a video to view"}
          </div>
        )}

        {/* Error banner */}
        {errorMessage && (
          <div
            style={{
              padding: "0.5rem 1rem",
              background: "#fff0f0",
              borderTop: "1px solid #ffcccc",
              color: "#cc0000",
              fontSize: "0.875rem",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexShrink: 0,
            }}
          >
            {errorMessage}
            <button
              onClick={() => setErrorMessage(null)}
              style={{ border: "none", background: "none", cursor: "pointer", color: "#cc0000", fontWeight: 700 }}
            >
              ✕
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
