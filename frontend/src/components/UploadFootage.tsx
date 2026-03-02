import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CameraParams } from "../App";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StepProgress {
  processed_frames: number;
}

type VideoMeta = {
  total_frames: number;
  fps?: number;
  steps?: Record<string, StepProgress>;
  current_step?: string | null;
  processing?: boolean;
} | null;


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_FPS = 30;

const STEP_LABELS: Record<string, string> = { inference: "Inference", lead: "Lead car", distances: "Distances", gps: "GPS" };
const STEP_ORDER = ["inference", "lead", "distances", "gps"];


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

// ---------------------------------------------------------------------------
// Subtitle component
// ---------------------------------------------------------------------------

interface ProgressSnapshot {
  frames: number;
  time: number;
  step: string;
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

function isAllStepsComplete(meta: VideoMeta): boolean {
  if (!meta || meta.total_frames <= 0 || !meta.steps) return false;
  return STEP_ORDER.every((step) => {
    const s = meta.steps?.[step];
    return s && s.processed_frames >= meta.total_frames;
  });
}

function VideoSubtitle({ filename, meta, progressRef }: { filename: string; meta: VideoMeta; progressRef: React.RefObject<Record<string, ProgressSnapshot>> }) {
  if (!meta || meta.total_frames === 0) {
    return <div style={{ color: "#aaa", fontSize: "0.78rem" }}>Not processed</div>;
  }
  if (isAllStepsComplete(meta)) {
    return <div style={{ color: "#4a0", fontSize: "0.78rem" }}>Processing complete</div>;
  }

  // Find the current step being processed
  const currentStep = meta.current_step
    ?? STEP_ORDER.find((s) => (meta.steps?.[s]?.processed_frames ?? 0) < meta.total_frames)
    ?? STEP_ORDER[0];
  const stepIdx = STEP_ORDER.indexOf(currentStep);
  const stepLabel = STEP_LABELS[currentStep] ?? currentStep;
  const processed = meta.steps?.[currentStep]?.processed_frames ?? 0;
  const pct = Math.round((processed / meta.total_frames) * 100);

  // Compute ETA from first observation of progress for current step
  let etaStr = "";
  const snap = progressRef.current?.[filename];
  if (snap && snap.step === currentStep && processed > snap.frames) {
    const elapsed = (Date.now() - snap.time) / 1000;
    const done = processed - snap.frames;
    const remaining = meta.total_frames - processed;
    const rate = done / elapsed;
    if (rate > 0) {
      etaStr = `ETA ${formatEta(remaining / rate)}`;
    }
  }

  return (
    <div style={{ color: "#888", fontSize: "0.78rem" }}>
      Step {stepIdx + 1}/{STEP_ORDER.length}: {stepLabel} {processed}/{meta.total_frames} ({pct}%){etaStr && `, ${etaStr}`}
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

export default function UploadFootage({ directory }: UploadFootageProps) {
  const [videos, setVideos] = useState<VideoEntry[]>([]);
  const [activeVideo, setActiveVideo] = useState<VideoEntry | null>(null);
  const [metas, setMetas] = useState<Record<string, VideoMeta>>({});
  const [processingStarted, setProcessingStarted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [fps, setFps] = useState(DEFAULT_FPS);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [debugImgUrl, setDebugImgUrl] = useState<string | null>(null);
  const [gpsInfo, setGpsInfo] = useState<{ gps: { lat: number; lon: number; datetime: string; speed_kmh: number } | null } | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const isScrubbing = useRef(false);
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
      if (m?.processing && m.current_step) {
        anyProcessing = true;
        const snap = progressSnapshots.current[fname];
        const currentFrames = m.steps?.[m.current_step]?.processed_frames ?? 0;
        // Reset snapshot if step changed or no snapshot yet
        if (!snap || snap.step !== m.current_step) {
          progressSnapshots.current[fname] = { frames: currentFrames, time: now, step: m.current_step };
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
        // Restore last active video from localStorage (dev only)
        const savedFilename = import.meta.env.DEV ? localStorage.getItem("gd_activeVideo") : null;
        const saved = savedFilename ? vids.find((v) => v.filename === savedFilename) : null;
        setActiveVideo(saved ?? vids[0] ?? null);
        fetchAllMeta(vids);
      })
      .catch((e) => setErrorMessage(String(e)));
  }, [directory, fetchAllMeta]);

  // ── Clear overlay when active video changes ─────────────────────────────
  useEffect(() => {
    if (!activeVideo) return;
    if (import.meta.env.DEV) localStorage.setItem("gd_activeVideo", activeVideo.filename);
    setOverlayUrl(null);
    setDebugImgUrl(null);
    setGpsInfo(null);
  }, [activeVideo]);

  // ── Set fps from metadata ─────────────────────────────────────────────
  useEffect(() => {
    if (!activeVideo) return;
    const meta = metas[activeVideo.filename];
    if (meta?.fps) setFps(meta.fps);
  }, [activeVideo, metas]);

  // ── Poll metadata while processing is active ────────────────────────────
  const allComplete = videos.length > 0 && videos.every((v) => isAllStepsComplete(metas[v.filename]));
  // Clear the flag only when everything finishes
  useEffect(() => {
    if (allComplete) setProcessingStarted(false);
  }, [allComplete]);
  useEffect(() => {
    if (!processingStarted || videos.length === 0) return;
    const id = setInterval(() => fetchAllMeta(videos), 1000);
    return () => clearInterval(id);
  }, [processingStarted, videos, fetchAllMeta]);

  // ── Update overlay URL for the given time ──────────────────────────────
  const updateOverlay = useCallback((time: number) => {
    if (!activeVideo) { setOverlayUrl(null); return; }
    const meta = metas[activeVideo.filename];
    const frameN = Math.floor(time * fps);
    const inferenceFrames = meta?.steps?.inference?.processed_frames ?? 0;
    if (!meta || inferenceFrames <= frameN) {
      setOverlayUrl(null);
      return;
    }
    setOverlayUrl(
      `/api/footage/overlay?filename=${encodeURIComponent(activeVideo.filename)}&directory=${encodeURIComponent(directory)}&frame=${frameN}`,
    );
  }, [activeVideo, fps, directory, metas]);

  // ── Fetch GPS info for the current frame ────────────────────────────────
  const updateGpsInfo = useCallback((time: number) => {
    if (!activeVideo) { setGpsInfo(null); return; }
    const meta = metas[activeVideo.filename];
    const gpsDone = meta && meta.total_frames > 0
      && (meta.steps?.gps?.processed_frames ?? 0) >= meta.total_frames;
    if (!gpsDone) { setGpsInfo(null); return; }
    const frameN = Math.floor(time * fps);
    fetch(`/api/footage/gps-info?filename=${encodeURIComponent(activeVideo.filename)}&directory=${encodeURIComponent(directory)}&frame=${frameN}`)
      .then((r) => r.ok ? r.json() : null)
      .then((data) => setGpsInfo(data ?? null))
      .catch(() => setGpsInfo(null));
  }, [activeVideo, fps, directory, metas]);

  // ── Lead timeline URLs (available when lead step is complete) ───────────
  const leadTimelineUrls = useMemo(() => {
    if (!activeVideo) return null;
    const meta = metas[activeVideo.filename];
    const leadDone = meta && meta.total_frames > 0
      && (meta.steps?.lead?.processed_frames ?? 0) >= meta.total_frames;
    if (!leadDone) return null;
    const qs = `filename=${encodeURIComponent(activeVideo.filename)}&directory=${encodeURIComponent(directory)}`;
    return { raw: `/api/footage/lead-timeline?${qs}`, smooth: `/api/footage/lead-timeline-smooth?${qs}` };
  }, [activeVideo, directory, metas]);

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
      if (m && m.total_frames > 0 && !isAllStepsComplete(m)) {
        // Find first incomplete step
        const step = STEP_ORDER.find((s) => (m.steps?.[s]?.processed_frames ?? 0) < m.total_frames) ?? STEP_ORDER[0];
        const frames = m.steps?.[step]?.processed_frames ?? 0;
        progressSnapshots.current[v.filename] = { frames, time: now, step };
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
            {/* Video + overlay */}
            <div style={{ flex: 1, background: "#1a1a1a", overflow: "hidden", position: "relative" }}>
              <video
                key={activeVideo.filename}
                ref={videoRef}
                src={activeVideo.video_url}
                style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
                onPlay={() => { setIsPlaying(true); setDebugImgUrl(null); }}
                onPause={() => {
                  setIsPlaying(false);
                  const vid = videoRef.current;
                  if (vid && import.meta.env.DEV) localStorage.setItem("gd_currentTime", String(vid.currentTime));
                }}
                onTimeUpdate={() => {
                  const vid = videoRef.current;
                  if (!vid) return;
                  if (!isScrubbing.current) setCurrentTime(vid.currentTime);
                  updateOverlay(vid.currentTime);
                  updateGpsInfo(vid.currentTime);
                }}
                onSeeked={() => {
                  const vid = videoRef.current;
                  if (vid) {
                    updateOverlay(vid.currentTime);
                    updateGpsInfo(vid.currentTime);
                    if (import.meta.env.DEV) localStorage.setItem("gd_currentTime", String(vid.currentTime));
                  }
                  setDebugImgUrl(null);
                }}
                onLoadedMetadata={() => {
                  const vid = videoRef.current;
                  if (vid) {
                    setDuration(vid.duration);
                    const savedTime = import.meta.env.DEV ? parseFloat(localStorage.getItem("gd_currentTime") ?? "0") : 0;
                    const t = Math.min(savedTime, vid.duration || 0);
                    vid.currentTime = t;
                    setCurrentTime(t);
                    setIsPlaying(false);
                    updateOverlay(t);
                    updateGpsInfo(t);
                  }
                }}
              />
              {debugImgUrl && (
                <img
                  src={debugImgUrl}
                  style={{
                    position: "absolute",
                    top: 0, left: 0,
                    width: "100%", height: "100%",
                    objectFit: "contain",
                    pointerEvents: "none",
                  }}
                />
              )}
              {!debugImgUrl && overlayUrl && (
                <img
                  src={overlayUrl}
                  style={{
                    position: "absolute",
                    top: 0, left: 0,
                    width: "100%", height: "100%",
                    objectFit: "contain",
                    pointerEvents: "none",
                  }}
                />
              )}
            </div>

            {/* Lead timelines */}
            {leadTimelineUrls && (
              <div style={{ padding: "0 1rem", flexShrink: 0, background: "#000" }}>
                <img
                  src={leadTimelineUrls.raw}
                  style={{
                    width: "100%",
                    height: "50px",
                    display: "block",
                    imageRendering: "pixelated",
                  }}
                />
                <img
                  src={leadTimelineUrls.smooth}
                  style={{
                    width: "100%",
                    height: "50px",
                    display: "block",
                    imageRendering: "pixelated",
                  }}
                />
              </div>
            )}

            {/* Controls */}
            <div
              style={{
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
                <button
                  onClick={async () => {
                    if (!activeVideo) return;
                    const frameN = Math.floor(currentTime * fps);
                    const res = await fetch(
                      `/api/footage/debug-frame?filename=${encodeURIComponent(activeVideo.filename)}&directory=${encodeURIComponent(directory)}&frame=${frameN}`,
                    );
                    if (!res.ok) return;
                    const blob = await res.blob();
                    if (debugImgUrl) URL.revokeObjectURL(debugImgUrl);
                    setDebugImgUrl(URL.createObjectURL(blob));
                  }}
                  style={btnStyle}
                >
                  Debug
                </button>
                <span
                  style={{
                    marginLeft: "0.5rem",
                    fontSize: "0.85rem",
                    color: "#555",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {formatTime(currentTime)} / {formatTime(duration)}
                  {" | "}Frame: {Math.floor(currentTime * fps)}
                  {gpsInfo?.gps && (
                    <>
                      {" | "}<a href={`https://www.google.com/maps?q=${gpsInfo.gps.lat},${gpsInfo.gps.lon}`} target="_blank" rel="noopener noreferrer">{gpsInfo.gps.lat.toFixed(5)}, {gpsInfo.gps.lon.toFixed(5)}</a>
                      {" | "}{gpsInfo.gps.speed_kmh.toFixed(0)} km/h
                      {" | "}{gpsInfo.gps.datetime}
                    </>
                  )}
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
