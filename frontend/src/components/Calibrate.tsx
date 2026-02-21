import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Detection {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  label: string;
}

interface CalibratedImage {
  filename: string;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  detections: Detection[] | null;
  isDetecting: boolean;
  selectedDetectionId: number | null;
  carWidthM: string;
  distanceM: string;
  estimatedDistanceM: number | null;
}

interface State {
  images: CalibratedImage[];
  activeImageIndex: number | null;
  solvedFovDeg: number | null;
  solvedPitchDeg: number | null;
  solvedHeightM: number | null;
  isSolving: boolean;
  errorMessage: string | null;
}

type Action =
  | { type: "SET_IMAGES"; images: CalibratedImage[] }
  | { type: "SELECT_IMAGE"; index: number }
  | { type: "DETECT_START"; index: number }
  | { type: "DETECT_SUCCESS"; index: number; detections: Detection[]; imageWidth: number; imageHeight: number }
  | { type: "SELECT_DETECTION"; index: number; detectionId: number | null }
  | { type: "SET_CAR_WIDTH"; index: number; value: string }
  | { type: "SET_DISTANCE"; index: number; value: string }
  | { type: "SOLVE_START" }
  | { type: "SOLVE_SUCCESS"; fovDeg: number; pitchDeg: number; heightM: number; perImage: { filename: string; estimated_distance_m: number }[] }
  | { type: "CLEAR_SOLVE" }
  | { type: "SET_ERROR"; message: string }
  | { type: "CLEAR_ERROR" };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SET_IMAGES":
      return { ...state, images: action.images, activeImageIndex: action.images.length > 0 ? 0 : null };

    case "SELECT_IMAGE":
      return { ...state, activeImageIndex: action.index };

    case "DETECT_START": {
      const images = [...state.images];
      images[action.index] = { ...images[action.index], isDetecting: true };
      return { ...state, images };
    }

    case "DETECT_SUCCESS": {
      const images = [...state.images];
      images[action.index] = {
        ...images[action.index],
        isDetecting: false,
        detections: action.detections,
        imageWidth: action.imageWidth,
        imageHeight: action.imageHeight,
      };
      return { ...state, images };
    }

    case "SELECT_DETECTION": {
      const images = [...state.images];
      images[action.index] = { ...images[action.index], selectedDetectionId: action.detectionId };
      return { ...state, images };
    }

    case "SET_CAR_WIDTH": {
      const images = [...state.images];
      images[action.index] = { ...images[action.index], carWidthM: action.value };
      return { ...state, images };
    }

    case "SET_DISTANCE": {
      const images = [...state.images];
      images[action.index] = { ...images[action.index], distanceM: action.value };
      return { ...state, images };
    }

    case "SOLVE_START":
      return { ...state, isSolving: true };

    case "SOLVE_SUCCESS": {
      const images = state.images.map((img) => {
        const match = action.perImage.find((p) => p.filename === img.filename);
        return match ? { ...img, estimatedDistanceM: match.estimated_distance_m } : img;
      });
      return {
        ...state,
        images,
        isSolving: false,
        solvedFovDeg: action.fovDeg,
        solvedPitchDeg: action.pitchDeg,
        solvedHeightM: action.heightM,
      };
    }

    case "CLEAR_SOLVE":
      return {
        ...state,
        images: state.images.map((img) => ({ ...img, estimatedDistanceM: null })),
        solvedFovDeg: null,
        solvedPitchDeg: null,
        solvedHeightM: null,
        isSolving: false,
      };

    case "SET_ERROR":
      return {
        ...state,
        errorMessage: action.message,
        isSolving: false,
        images: state.images.map((img) => ({ ...img, isDetecting: false })),
      };

    case "CLEAR_ERROR":
      return { ...state, errorMessage: null };

    default:
      return state;
  }
}

const initialState: State = {
  images: [],
  activeImageIndex: null,
  solvedFovDeg: null,
  solvedPitchDeg: null,
  solvedHeightM: null,
  isSolving: false,
  errorMessage: null,
};

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiDetect(directory: string, filename: string) {
  const res = await fetch("/api/calibrate/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ directory, filename }),
  });
  if (!res.ok) throw new Error(`Detection failed: ${res.statusText}`);
  return res.json();
}

async function apiSaveAnnotation(
  directory: string,
  filename: string,
  selectedDetectionId: number | null,
  carWidthM: number | null,
  distanceM: number | null,
) {
  await fetch("/api/calibrate/annotation", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      directory,
      filename,
      selected_detection_id: selectedDetectionId,
      car_width_m: carWidthM,
      distance_m: distanceM,
    }),
  });
}

async function apiSolve(directory: string) {
  const res = await fetch("/api/calibrate/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ directory }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error((err as { detail?: string }).detail ?? res.statusText);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------

function drawDetections(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  detections: Detection[],
  selectedId: number | null,
  imageWidth: number,
  imageHeight: number,
) {
  const ctx = canvas.getContext("2d")!;
  const containerW = img.clientWidth;
  const containerH = img.clientHeight;

  // object-fit: contain letterboxing
  const scale = Math.min(containerW / imageWidth, containerH / imageHeight);
  const renderW = imageWidth * scale;
  const renderH = imageHeight * scale;
  const offsetX = (containerW - renderW) / 2;
  const offsetY = (containerH - renderH) / 2;

  canvas.width = containerW;
  canvas.height = containerH;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const det of detections) {
    const x = det.x1 * scale + offsetX;
    const y = det.y1 * scale + offsetY;
    const w = (det.x2 - det.x1) * scale;
    const h = (det.y2 - det.y1) * scale;
    const selected = det.id === selectedId;

    const lw = selected ? 3 : 2;
    ctx.strokeStyle = selected ? "#ff6600" : "#00cc44";
    ctx.lineWidth = lw;
    ctx.strokeRect(x, y, w, h);

    const text = `${det.label} (${Math.round(det.confidence * 100)}%)`;
    const color = selected ? "#ff6600" : "#00cc44";
    ctx.font = "bold 12px system-ui";
    const textW = ctx.measureText(text).width;
    const pad = 3;
    const bgH = 14;
    const bgLeft = x - lw / 2;
    ctx.fillStyle = color;
    ctx.fillRect(bgLeft, y - bgH, textW + pad * 2, bgH);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, bgLeft + pad, y - 3);
  }
}

function hitTest(
  clickX: number,
  clickY: number,
  containerW: number,
  containerH: number,
  detections: Detection[],
  imageWidth: number,
  imageHeight: number,
): number | null {
  const scale = Math.min(containerW / imageWidth, containerH / imageHeight);
  const renderW = imageWidth * scale;
  const renderH = imageHeight * scale;
  const offsetX = (containerW - renderW) / 2;
  const offsetY = (containerH - renderH) / 2;

  const imgX = (clickX - offsetX) / scale;
  const imgY = (clickY - offsetY) / scale;

  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  for (const det of sorted) {
    if (imgX >= det.x1 && imgX <= det.x2 && imgY >= det.y1 && imgY <= det.y2) {
      return det.id;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Sidebar image subtitle helper
// ---------------------------------------------------------------------------

function ImageSubtitle({ img }: { img: CalibratedImage }) {
  if (img.selectedDetectionId === null) {
    return <div style={{ color: "#aaa", fontSize: "0.78rem" }}>Car not selected</div>;
  }
  if (!img.carWidthM) {
    return <div style={{ color: "#aaa", fontSize: "0.78rem" }}>Car width not entered</div>;
  }
  if (!img.distanceM) {
    return <div style={{ color: "#aaa", fontSize: "0.78rem" }}>Car distance not entered</div>;
  }

  const trueD = parseFloat(img.distanceM);
  const estD = img.estimatedDistanceM;
  const deviation = estD != null && trueD > 0
    ? Math.round(Math.abs(estD - trueD) / trueD * 100)
    : null;

  return (
    <div style={{ color: "#555", fontSize: "0.78rem", lineHeight: 1.55 }}>
      <div>
        True = {img.distanceM} m
        {estD != null && <> | Est. = {estD} m</>}
        {deviation != null && <> | Δ = {deviation}%</>}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface CalibrateProps {
  directory: string;
}

export default function Calibrate({ directory }: CalibrateProps) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const annotationTimers = useRef<Record<number, ReturnType<typeof setTimeout>>>({});
  const solveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load images on mount
  useEffect(() => {
    fetch("/api/calibrate/open-directory", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        const images: CalibratedImage[] = (data.images ?? []).map(
          (img: Record<string, unknown>) => ({
            filename: img.filename as string,
            imageUrl: img.image_url as string,
            imageWidth: (img.image_width as number) ?? 0,
            imageHeight: (img.image_height as number) ?? 0,
            detections: (img.detections as Detection[] | null) ?? null,
            isDetecting: false,
            selectedDetectionId: (img.selected_detection_id as number | null) ?? null,
            carWidthM: img.car_width_m != null ? String(img.car_width_m) : "2.0",
            distanceM: img.distance_m != null ? String(img.distance_m) : "",
            estimatedDistanceM: null,
          }),
        );
        dispatch({ type: "SET_IMAGES", images });
      })
      .catch((e) => dispatch({ type: "SET_ERROR", message: String(e) }));
  }, [directory]);

  // Auto-detect when switching to an image with no detections yet
  useEffect(() => {
    const idx = state.activeImageIndex;
    if (idx === null) return;
    const img = state.images[idx];
    if (!img || img.detections !== null || img.isDetecting) return;

    dispatch({ type: "DETECT_START", index: idx });
    apiDetect(directory, img.filename)
      .then((data) => {
        dispatch({
          type: "DETECT_SUCCESS",
          index: idx,
          detections: data.detections ?? [],
          imageWidth: data.image_width ?? img.imageWidth,
          imageHeight: data.image_height ?? img.imageHeight,
        });
      })
      .catch((e) => dispatch({ type: "SET_ERROR", message: String(e) }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.activeImageIndex, directory]);

  // Auto-solve: fire when annotation data changes
  const annotationFingerprint = useMemo(
    () => state.images.map((img) => `${img.selectedDetectionId}|${img.carWidthM}|${img.distanceM}`).join(","),
    [state.images],
  );

  useEffect(() => {
    if (solveTimer.current) {
      clearTimeout(solveTimer.current);
      solveTimer.current = null;
    }

    const complete = state.images.filter(
      (img) => img.selectedDetectionId !== null && img.carWidthM && img.distanceM,
    );

    if (complete.length < 2) {
      dispatch({ type: "CLEAR_SOLVE" });
      return;
    }

    // Wait for annotation saves (500ms debounce) to flush before solving
    solveTimer.current = setTimeout(async () => {
      solveTimer.current = null;
      dispatch({ type: "SOLVE_START" });
      try {
        const result = await apiSolve(directory);
        dispatch({
          type: "SOLVE_SUCCESS",
          fovDeg: result.fov_degrees,
          pitchDeg: result.pitch_degrees,
          heightM: result.camera_height_m,
          perImage: result.per_image,
        });
      } catch {
        dispatch({ type: "CLEAR_SOLVE" });
      }
    }, 1200);

    return () => {
      if (solveTimer.current) {
        clearTimeout(solveTimer.current);
        solveTimer.current = null;
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [annotationFingerprint, directory]);

  // Redraw canvas
  const redraw = useCallback(() => {
    const idx = state.activeImageIndex;
    if (idx === null) return;
    const img = state.images[idx];
    if (!img?.detections || !imgRef.current || !canvasRef.current) return;
    if (img.imageWidth === 0 || img.imageHeight === 0) return;
    drawDetections(
      canvasRef.current,
      imgRef.current,
      img.detections,
      img.selectedDetectionId,
      img.imageWidth,
      img.imageHeight,
    );
  }, [state.activeImageIndex, state.images]);

  useEffect(() => { redraw(); }, [redraw]);

  useEffect(() => {
    if (!imgRef.current) return;
    const ro = new ResizeObserver(redraw);
    ro.observe(imgRef.current);
    return () => ro.disconnect();
  }, [redraw]);

  // Debounced annotation save
  const scheduleAnnotationSave = useCallback(
    (index: number, detectionId: number | null, carWidth: string, distance: string) => {
      if (annotationTimers.current[index]) clearTimeout(annotationTimers.current[index]);
      annotationTimers.current[index] = setTimeout(() => {
        const img = state.images[index];
        if (!img) return;
        apiSaveAnnotation(
          directory,
          img.filename,
          detectionId,
          carWidth ? parseFloat(carWidth) : null,
          distance ? parseFloat(distance) : null,
        );
      }, 500);
    },
    [directory, state.images],
  );

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const idx = state.activeImageIndex;
      if (idx === null || !canvasRef.current) return;
      const img = state.images[idx];
      if (!img?.detections) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const scaleX = canvasRef.current.width / rect.width;
      const scaleY = canvasRef.current.height / rect.height;
      const clickX = (e.clientX - rect.left) * scaleX;
      const clickY = (e.clientY - rect.top) * scaleY;

      const detId = hitTest(
        clickX, clickY,
        canvasRef.current.width, canvasRef.current.height,
        img.detections, img.imageWidth, img.imageHeight,
      );
      dispatch({ type: "SELECT_DETECTION", index: idx, detectionId: detId });
      scheduleAnnotationSave(idx, detId, img.carWidthM, img.distanceM);
    },
    [state.activeImageIndex, state.images, scheduleAnnotationSave],
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const activeImage = state.activeImageIndex !== null ? state.images[state.activeImageIndex] : null;

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
          {state.images.length === 0 && (
            <div style={{ color: "#999", fontSize: "0.85rem", padding: "0.5rem" }}>
              No images found in directory
            </div>
          )}
          {state.images.map((img, idx) => (
            <div
              key={img.filename}
              onClick={() => dispatch({ type: "SELECT_IMAGE", index: idx })}
              style={{
                padding: "0.4rem 0.6rem",
                marginBottom: "2px",
                cursor: "pointer",
                borderRadius: "4px",
                background: state.activeImageIndex === idx ? "#e8f0fe" : "transparent",
                fontSize: "0.875rem",
              }}
            >
              <div
                style={{
                  fontWeight: state.activeImageIndex === idx ? 600 : 400,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  color: img.selectedDetectionId !== null && img.carWidthM && img.distanceM ? "#000" : "#555",
                }}
              >
                {img.isDetecting ? "⏳ " : ""}{img.filename}
              </div>
              <ImageSubtitle img={img} />
            </div>
          ))}
        </div>

        {/* Camera parameters — always shown */}
        <div style={{ borderTop: "1px solid #ddd", padding: "0.75rem", flexShrink: 0 }}>
          <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "#888", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: "0.4rem" }}>
            Camera parameters
          </div>
          {state.isSolving ? (
            <div style={{ fontSize: "0.82rem", color: "#999" }}>Calculating…</div>
          ) : state.solvedFovDeg !== null ? (
            <div style={{ fontSize: "0.82rem", lineHeight: 1.8, color: "#222" }}>
              <div>Field of view: <strong>{state.solvedFovDeg}°</strong></div>
              <div>Height: <strong>{state.solvedHeightM} m</strong></div>
              <div>Pitch: <strong>{state.solvedPitchDeg}°</strong></div>
            </div>
          ) : (
            <div style={{ fontSize: "0.8rem", color: "#aaa", lineHeight: 1.5 }}>
              Annotate at least two images to estimate camera parameters.
            </div>
          )}
        </div>
      </div>

      {/* ── RIGHT PANEL ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
        {activeImage ? (
          <>
            {/* Image + overlay */}
            <div
              style={{
                flex: 1,
                position: "relative",
                overflow: "hidden",
                background: "#1a1a1a",
              }}
            >
              {activeImage.isDetecting && (
                <div
                  style={{
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                    background: "rgba(0,0,0,0.7)",
                    color: "#fff",
                    padding: "0.6rem 1.25rem",
                    borderRadius: "6px",
                    zIndex: 10,
                    fontSize: "0.9rem",
                  }}
                >
                  Detecting vehicles…
                </div>
              )}
              <img
                ref={imgRef}
                src={activeImage.imageUrl}
                alt={activeImage.filename}
                onLoad={redraw}
                style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
              />
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  cursor:
                    activeImage.detections && activeImage.detections.length > 0
                      ? "crosshair"
                      : "default",
                }}
              />
            </div>

            {/* Controls */}
            <div
              style={{
                borderTop: "1px solid #ddd",
                padding: "0.6rem 1rem",
                display: "flex",
                alignItems: "center",
                gap: "1.25rem",
                flexShrink: 0,
                background: "#fff",
              }}
            >
              <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.9rem" }}>
                Car width (m):
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={activeImage.carWidthM}
                  onChange={(e) => {
                    const idx = state.activeImageIndex!;
                    dispatch({ type: "SET_CAR_WIDTH", index: idx, value: e.target.value });
                    scheduleAnnotationSave(
                      idx, activeImage.selectedDetectionId, e.target.value, activeImage.distanceM,
                    );
                  }}
                  style={{ width: "68px" }}
                />
              </label>

              <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.9rem" }}>
                Distance (m):
                <input
                  type="number"
                  step="1"
                  min="0"
                  value={activeImage.distanceM}
                  onChange={(e) => {
                    const idx = state.activeImageIndex!;
                    dispatch({ type: "SET_DISTANCE", index: idx, value: e.target.value });
                    scheduleAnnotationSave(
                      idx, activeImage.selectedDetectionId, activeImage.carWidthM, e.target.value,
                    );
                  }}
                  style={{ width: "68px" }}
                />
              </label>

              {activeImage.detections !== null && activeImage.selectedDetectionId === null && (
                <span style={{ color: "#888", fontSize: "0.85rem" }}>Click a vehicle to select it</span>
              )}
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
            {state.images.length === 0 ? "No images in directory" : "Select an image"}
          </div>
        )}

        {/* Error banner */}
        {state.errorMessage && (
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
            {state.errorMessage}
            <button
              onClick={() => dispatch({ type: "CLEAR_ERROR" })}
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
