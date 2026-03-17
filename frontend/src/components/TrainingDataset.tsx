import { useCallback, useEffect, useRef, useState } from "react";

interface TrainingDatasetProps {
  directory: string;
}

interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface RandomFrame {
  image_base64: string;
  video_filename: string;
  frame_number: number;
  width: number;
  height: number;
  detections: Detection[];
}

interface Stats {
  with_car: number;
  without_car: number;
}

interface EvalResult {
  image: string;
  match: boolean;
  detail: string;
}

interface EvalResults {
  total: number;
  matches: number;
  mismatches: number;
  results: EvalResult[];
}

type Phase = "empty" | "loading" | "select-car" | "mark-rear" | "saving";

export default function TrainingDataset({ directory }: TrainingDatasetProps) {
  const [images, setImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("empty");
  const [frame, setFrame] = useState<RandomFrame | null>(null);
  const [selectedBbox, setSelectedBbox] = useState<Detection | null>(null);
  const [mouseX, setMouseX] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats>({ with_car: 0, without_car: 0 });
  const [evaluating, setEvaluating] = useState(false);
  const [evalResults, setEvalResults] = useState<EvalResults | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // Track display scaling: image coords = (canvas coords - offset) / scale
  const scaleRef = useRef({ scale: 1, offsetX: 0, offsetY: 0, displayW: 0, displayH: 0 });

  const refreshImages = useCallback(() => {
    fetch(`/api/footage/dataset-images?directory=${encodeURIComponent(directory)}`)
      .then((r) => r.json())
      .then((data) => setImages(data.images ?? []));
  }, [directory]);

  const refreshStats = useCallback(() => {
    fetch(`/api/footage/dataset-stats?directory=${encodeURIComponent(directory)}`)
      .then((r) => r.json())
      .then((data) => setStats(data));
  }, [directory]);

  const runEvaluate = useCallback(async () => {
    setEvaluating(true);
    setEvalResults(null);
    try {
      const res = await fetch(`/api/footage/dataset-evaluate?directory=${encodeURIComponent(directory)}`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(res.statusText);
      const data: EvalResults = await res.json();
      setEvalResults(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setEvaluating(false);
    }
  }, [directory]);

  useEffect(() => {
    refreshImages();
    refreshStats();
  }, [refreshImages, refreshStats]);

  const fetchRandomFrame = useCallback(async () => {
    setPhase("loading");
    setFrame(null);
    setSelectedBbox(null);
    setMouseX(null);
    setError(null);
    setSelectedImage(null);
    try {
      const res = await fetch(`/api/footage/dataset-random-frame?directory=${encodeURIComponent(directory)}`, {
        method: "POST",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || res.statusText);
      }
      const data: RandomFrame = await res.json();
      setFrame(data);
      setPhase("select-car");
    } catch (e) {
      setError(String(e));
      setPhase("empty");
    }
  }, [directory]);

  const saveAnnotation = useCallback(
    async (bbox: Detection, rearCenterX: number) => {
      if (!frame) return;
      setPhase("saving");
      try {
        const params = new URLSearchParams({
          directory,
          video_filename: frame.video_filename,
          frame_number: String(frame.frame_number),
          bbox_x1: String(bbox.x1),
          bbox_y1: String(bbox.y1),
          bbox_x2: String(bbox.x2),
          bbox_y2: String(bbox.y2),
          rear_center_x: String(rearCenterX),
        });
        const res = await fetch(`/api/footage/dataset-save?${params}`, { method: "POST" });
        if (!res.ok) throw new Error(res.statusText);
        refreshImages();
        refreshStats();
        // Immediately fetch next frame
        fetchRandomFrame();
      } catch (e) {
        setError(String(e));
        setPhase("select-car");
      }
    },
    [frame, directory, refreshImages, refreshStats, fetchRandomFrame],
  );

  const saveNoCar = useCallback(async () => {
    if (!frame) return;
    setPhase("saving");
    try {
      const params = new URLSearchParams({
        directory,
        video_filename: frame.video_filename,
        frame_number: String(frame.frame_number),
      });
      const res = await fetch(`/api/footage/dataset-save?${params}`, { method: "POST" });
      if (!res.ok) throw new Error(res.statusText);
      refreshImages();
      refreshStats();
      fetchRandomFrame();
    } catch (e) {
      setError(String(e));
      setPhase("select-car");
    }
  }, [frame, directory, refreshImages, refreshStats, fetchRandomFrame]);

  // Escape / Enter key handler
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && (phase === "select-car" || phase === "mark-rear")) {
        fetchRandomFrame();
      } else if (e.key === "Enter" && phase === "select-car") {
        saveNoCar();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [phase, fetchRandomFrame, saveNoCar]);

  // Compute scale/offset for fitting image in container
  const computeLayout = useCallback(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas || !frame) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    canvas.width = cw;
    canvas.height = ch;

    const scale = Math.min(cw / frame.width, ch / frame.height);
    const displayW = frame.width * scale;
    const displayH = frame.height * scale;
    const offsetX = (cw - displayW) / 2;
    const offsetY = (ch - displayH) / 2;
    scaleRef.current = { scale, offsetX, offsetY, displayW, displayH };
  }, [frame]);

  // Load image when frame changes
  useEffect(() => {
    if (!frame) return;
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      computeLayout();
      drawCanvas();
    };
    img.src = `data:image/jpeg;base64,${frame.image_base64}`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  // Resize handler
  useEffect(() => {
    const onResize = () => {
      computeLayout();
      drawCanvas();
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [computeLayout]);

  // Convert canvas coords to image coords
  const canvasToImage = useCallback((cx: number, cy: number) => {
    const { scale, offsetX, offsetY } = scaleRef.current;
    return { ix: (cx - offsetX) / scale, iy: (cy - offsetY) / scale };
  }, []);

  // Convert image coords to canvas coords
  const imageToCanvas = useCallback((ix: number, iy: number) => {
    const { scale, offsetX, offsetY } = scaleRef.current;
    return { cx: ix * scale + offsetX, cy: iy * scale + offsetY };
  }, []);

  // Draw everything on canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !frame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { scale: _scale, offsetX, offsetY, displayW, displayH } = scaleRef.current;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, offsetX, offsetY, displayW, displayH);

    // Draw detection bboxes
    for (const det of frame.detections) {
      const isSelected = selectedBbox && det.x1 === selectedBbox.x1 && det.y1 === selectedBbox.y1 && det.x2 === selectedBbox.x2 && det.y2 === selectedBbox.y2;
      const tl = imageToCanvas(det.x1, det.y1);
      const br = imageToCanvas(det.x2, det.y2);
      const w = br.cx - tl.cx;
      const h = br.cy - tl.cy;

      if (isSelected) {
        ctx.fillStyle = "rgba(255, 0, 0, 0.15)";
        ctx.fillRect(tl.cx, tl.cy, w, h);
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
      } else {
        ctx.fillStyle = "rgba(0, 120, 255, 0.1)";
        ctx.fillRect(tl.cx, tl.cy, w, h);
        ctx.strokeStyle = "rgba(0, 120, 255, 0.7)";
        ctx.lineWidth = 1.5;
      }
      ctx.setLineDash([]);
      ctx.strokeRect(tl.cx, tl.cy, w, h);
    }

    // Draw vertical dotted cursor line in selected bbox during mark-rear phase
    if (phase === "mark-rear" && selectedBbox && mouseX !== null) {
      const tl = imageToCanvas(selectedBbox.x1, selectedBbox.y1);
      const br = imageToCanvas(selectedBbox.x2, selectedBbox.y2);
      // Clamp mouseX to bbox bounds (in canvas coords)
      const clampedCx = Math.max(tl.cx, Math.min(br.cx, mouseX));

      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1.5;
      ctx.moveTo(clampedCx, tl.cy);
      ctx.lineTo(clampedCx, br.cy);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Cursor hint
    if (phase === "select-car") {
      ctx.fillStyle = "rgba(255,255,255,0.8)";
      ctx.font = "14px system-ui";
      ctx.fillText("Click a car to select · Enter = no leading car · Esc = skip", offsetX + 10, offsetY + displayH - 10);
    } else if (phase === "mark-rear") {
      ctx.fillStyle = "rgba(255,255,255,0.8)";
      ctx.font = "14px system-ui";
      ctx.fillText("Click to mark the rear center", offsetX + 10, offsetY + displayH - 10);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame, selectedBbox, mouseX, phase, imageToCanvas]);

  // Redraw when state changes
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!frame) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const { ix, iy } = canvasToImage(cx, cy);

      if (phase === "select-car") {
        // Find which bbox was clicked
        for (const det of frame.detections) {
          if (ix >= det.x1 && ix <= det.x2 && iy >= det.y1 && iy <= det.y2) {
            setSelectedBbox(det);
            setPhase("mark-rear");
            return;
          }
        }
      } else if (phase === "mark-rear" && selectedBbox) {
        // Clamp click to selected bbox
        const rearCenterX = Math.max(selectedBbox.x1, Math.min(selectedBbox.x2, Math.round(ix)));
        saveAnnotation(selectedBbox, rearCenterX);
      }
    },
    [frame, phase, selectedBbox, canvasToImage, saveAnnotation],
  );

  const handleCanvasMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (phase !== "mark-rear") return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      setMouseX(e.clientX - rect.left);
    },
    [phase],
  );

  // View a saved image from the list
  const viewSavedImage = useCallback(
    (name: string) => {
      setSelectedImage(name);
      setFrame(null);
      setSelectedBbox(null);
      setPhase("empty");

      // Load saved image onto canvas
      const img = new window.Image();
      img.onload = () => {
        imgRef.current = img;
        const container = containerRef.current;
        const canvas = canvasRef.current;
        if (!container || !canvas) return;

        const cw = container.clientWidth;
        const ch = container.clientHeight;
        canvas.width = cw;
        canvas.height = ch;

        const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
        const displayW = img.naturalWidth * scale;
        const displayH = img.naturalHeight * scale;
        const offsetX = (cw - displayW) / 2;
        const offsetY = (ch - displayH) / 2;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.clearRect(0, 0, cw, ch);
        ctx.drawImage(img, offsetX, offsetY, displayW, displayH);
      };
      img.src = `/api/footage/dataset-debug-image?filename=${encodeURIComponent(name)}&directory=${encodeURIComponent(directory)}&_t=${Date.now()}`;
    },
    [directory],
  );

  const showingAnnotation = phase === "select-car" || phase === "mark-rear" || phase === "loading" || phase === "saving";

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      {/* Left panel: image list */}
      <div
        style={{
          width: "300px",
          flexShrink: 0,
          borderRight: "1px solid #ddd",
          overflowY: "auto",
          padding: "0.5rem",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <button
          onClick={fetchRandomFrame}
          disabled={phase === "loading" || phase === "saving"}
          style={{
            margin: "0.5rem",
            padding: "0.5rem 1rem",
            background: "#0066cc",
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            cursor: phase === "loading" || phase === "saving" ? "default" : "pointer",
            fontSize: "0.9rem",
            opacity: phase === "loading" || phase === "saving" ? 0.6 : 1,
          }}
        >
          {phase === "loading" ? "Loading..." : "Add image"}
        </button>

        {(stats.with_car > 0 || stats.without_car > 0) && (
          <div style={{ padding: "0.25rem 0.5rem", fontSize: "0.8rem", color: "#666", borderBottom: "1px solid #eee", marginBottom: "0.25rem" }}>
            {stats.with_car} with leading car, {stats.without_car} without
          </div>
        )}

        <button
          onClick={runEvaluate}
          disabled={evaluating || stats.with_car + stats.without_car === 0}
          style={{
            margin: "0 0.5rem 0.5rem",
            padding: "0.5rem 1rem",
            background: "#228B22",
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            cursor: evaluating || stats.with_car + stats.without_car === 0 ? "default" : "pointer",
            fontSize: "0.9rem",
            opacity: evaluating || stats.with_car + stats.without_car === 0 ? 0.6 : 1,
          }}
        >
          {evaluating ? "Evaluating..." : "Evaluate"}
        </button>

        {evalResults && (
          <div style={{ padding: "0.25rem 0.5rem", fontSize: "0.8rem", borderBottom: "1px solid #eee", marginBottom: "0.25rem" }}>
            <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
              {evalResults.matches}/{evalResults.total} match ({evalResults.mismatches} mismatch)
            </div>
            {evalResults.results.filter((r) => !r.match).map((r) => (
              <div
                key={r.image}
                onClick={() => viewSavedImage(r.image)}
                style={{ color: "#cc0000", cursor: "pointer", padding: "2px 0" }}
              >
                {r.image}: {r.detail}
              </div>
            ))}
          </div>
        )}

        {images.length === 0 && (
          <div style={{ color: "#999", fontSize: "0.85rem", padding: "0.5rem" }}>
            No images in Dataset directory
          </div>
        )}
        {images.map((name) => (
          <div
            key={name}
            onClick={() => viewSavedImage(name)}
            style={{
              padding: "0.4rem 0.6rem",
              marginBottom: "2px",
              cursor: "pointer",
              borderRadius: "4px",
              background: selectedImage === name && !showingAnnotation ? "#e8f0fe" : "transparent",
              fontSize: "0.875rem",
            }}
          >
            <div
              style={{
                fontWeight: selectedImage === name && !showingAnnotation ? 600 : 400,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                color: "#000",
              }}
            >
              {name}
            </div>
          </div>
        ))}
      </div>

      {/* Right panel: annotation area */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          background: "#1a1a1a",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {phase === "empty" && !selectedImage && (
          <div style={{ color: "#999", fontSize: "1rem" }}>
            {error ? (
              <span style={{ color: "#ff6666" }}>{error}</span>
            ) : (
              'Click "Add image" to start annotating'
            )}
          </div>
        )}

        {(showingAnnotation || selectedImage) && (
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMove}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              cursor:
                phase === "select-car"
                  ? "crosshair"
                  : phase === "mark-rear"
                    ? "crosshair"
                    : "default",
            }}
          />
        )}

        {phase === "loading" && (
          <div style={{ color: "#999", position: "absolute" }}>Loading random frame...</div>
        )}
        {phase === "saving" && (
          <div style={{ color: "#999", position: "absolute" }}>Saving...</div>
        )}
      </div>
    </div>
  );
}
