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

type VideoMeta = { total_frames: number; processed_frames: number; fps?: number } | null;

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

const btnStyle: React.CSSProperties = {
  padding: "0.3rem 0.7rem",
  border: "1px solid #ccc",
  borderRadius: "4px",
  background: "#f5f5f5",
  cursor: "pointer",
  fontSize: "0.85rem",
  fontFamily: "system-ui",
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
// Subtitle component
// ---------------------------------------------------------------------------

function VideoSubtitle({ meta }: { meta: VideoMeta }) {
  if (!meta || meta.total_frames === 0) {
    return <div style={{ color: "#aaa", fontSize: "0.78rem" }}>Not processed</div>;
  }
  if (meta.processed_frames >= meta.total_frames) {
    return <div style={{ color: "#4a0", fontSize: "0.78rem" }}>Processing complete</div>;
  }
  const pct = Math.round((meta.processed_frames / meta.total_frames) * 100);
  return (
    <div style={{ color: "#888", fontSize: "0.78rem" }}>
      Processed {meta.processed_frames}/{meta.total_frames} frames ({pct}%)
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface UploadFootageProps {
  cameraParams: CameraParams;
}

export default function UploadFootage({ cameraParams }: UploadFootageProps) {
  const [videos, setVideos] = useState<string[]>([]);
  const [activeVideo, setActiveVideo] = useState<string | null>(null);
  const [metas, setMetas] = useState<Record<string, VideoMeta>>({});
  const [isStarting, setIsStarting] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [fps, setFps] = useState(DEFAULT_FPS);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bevCanvasRef = useRef<HTMLCanvasElement>(null);
  const isScrubbing = useRef(false);
  // Cache batch data: batchIndex → FrameEntry[]
  const batchCache = useRef<Record<number, FrameEntry[]>>({});
  const frameDataRef = useRef<FrameEntry | null>(null);

  // ── Fetch metadata for all videos ──────────────────────────────────────
  const fetchAllMeta = useCallback(async (videoList: string[]) => {
    const entries = await Promise.all(
      videoList.map(async (filename) => {
        const r = await fetch(`/api/footage/metadata?filename=${encodeURIComponent(filename)}`);
        const data = r.ok ? await r.json() : null;
        return [filename, data] as [string, VideoMeta];
      }),
    );
    setMetas(Object.fromEntries(entries));
  }, []);

  // ── Load video list on mount ────────────────────────────────────────────
  useEffect(() => {
    fetch("/api/footage/list")
      .then((r) => r.json())
      .then((data) => {
        const vids: string[] = data.videos ?? [];
        setVideos(vids);
        if (vids.length > 0) setActiveVideo(vids[0]);
        fetchAllMeta(vids);
      })
      .catch((e) => setErrorMessage(String(e)));
  }, [fetchAllMeta]);

  // ── Set fps from metadata when a video is selected ─────────────────────
  useEffect(() => {
    if (!activeVideo) return;
    const meta = metas[activeVideo];
    if (meta?.fps) setFps(meta.fps);
    // Clear stale frame data when switching video
    batchCache.current = {};
    frameDataRef.current = null;
  }, [activeVideo, metas]);

  // ── Poll metadata while any video is processing ─────────────────────────
  const anyProcessing = Object.values(metas).some(
    (m) => m !== null && m.total_frames > 0 && m.processed_frames < m.total_frames,
  );
  useEffect(() => {
    if (!anyProcessing || videos.length === 0) return;
    const id = setInterval(() => fetchAllMeta(videos), 2000);
    return () => clearInterval(id);
  }, [anyProcessing, videos, fetchAllMeta]);

  // ── Canvas drawing ──────────────────────────────────────────────────────
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const bevCanvas = bevCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    // Sync canvas resolution to its CSS size
    if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }
    drawOverlay(canvas, video, frameDataRef.current, cameraParams);
    // BEV
    if (bevCanvas) {
      if (bevCanvas.width !== bevCanvas.clientWidth || bevCanvas.height !== bevCanvas.clientHeight) {
        bevCanvas.width = bevCanvas.clientWidth;
        bevCanvas.height = bevCanvas.clientHeight;
      }
      drawBEV(bevCanvas, frameDataRef.current, cameraParams, video.videoWidth, video.videoHeight);
    }
  }, [cameraParams]);

  // Redraw when either canvas is resized
  useEffect(() => {
    const canvas = canvasRef.current;
    const bevCanvas = bevCanvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);
    if (bevCanvas) ro.observe(bevCanvas);
    return () => ro.disconnect();
  }, [redraw]);

  // ── Load frame data for the given time and redraw ───────────────────────
  const loadFrameAndDraw = useCallback(async (time: number) => {
    const video = activeVideo;
    if (!video) return;

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
        `/api/footage/frames?filename=${encodeURIComponent(video)}&batch=${batchIdx}`,
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
  const allComplete = videos.length > 0 && videos.every((v) => {
    const m = metas[v];
    return m !== null && m !== undefined && m.total_frames > 0 && m.processed_frames >= m.total_frames;
  });

  const handleStartProcessing = async () => {
    setIsStarting(true);
    try {
      await fetch("/api/footage/start-processing", { method: "POST" });
      await fetchAllMeta(videos);
    } catch (e) {
      setErrorMessage(String(e));
    } finally {
      setIsStarting(false);
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
          {videos.map((filename) => (
            <div
              key={filename}
              onClick={() => setActiveVideo(filename)}
              style={{
                padding: "0.4rem 0.6rem",
                marginBottom: "2px",
                cursor: "pointer",
                borderRadius: "4px",
                background: activeVideo === filename ? "#e8f0fe" : "transparent",
                fontSize: "0.875rem",
              }}
            >
              <div
                style={{
                  fontWeight: activeVideo === filename ? 600 : 400,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  color: "#000",
                }}
              >
                {filename}
              </div>
              <VideoSubtitle meta={metas[filename] ?? null} />
            </div>
          ))}
        </div>

        {/* Start processing button */}
        <div style={{ borderTop: "1px solid #ddd", padding: "0.75rem", flexShrink: 0 }}>
          <button
            onClick={handleStartProcessing}
            disabled={isStarting || allComplete || anyProcessing || videos.length === 0}
            style={{
              width: "100%",
              padding: "0.5rem",
              border: "1px solid #ccc",
              borderRadius: "4px",
              background: allComplete || anyProcessing ? "#f0f0f0" : "#0066cc",
              color: allComplete || anyProcessing ? "#999" : "#fff",
              cursor: isStarting || allComplete || anyProcessing ? "default" : "pointer",
              fontSize: "0.9rem",
              fontFamily: "system-ui",
            }}
          >
            {isStarting
              ? "Starting…"
              : anyProcessing
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
                  key={activeVideo}
                  ref={videoRef}
                  src={`/api/footage/video?filename=${encodeURIComponent(activeVideo)}`}
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
              {/* Bird's Eye View */}
              <div style={{ width: "220px", flexShrink: 0, borderLeft: "1px solid #333" }}>
                <canvas
                  ref={bevCanvasRef}
                  style={{ width: "100%", height: "100%", display: "block" }}
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
            {videos.length === 0 ? "No videos in directory" : "Select a video"}
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
