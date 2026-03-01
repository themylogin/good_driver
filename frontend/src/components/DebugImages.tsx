import { useCallback, useEffect, useState } from "react";

interface DebugImagesProps {
  directory: string;
}

export default function DebugImages({ directory }: DebugImagesProps) {
  const [images, setImages] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [debugUrl, setDebugUrl] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/footage/debug-images?directory=${encodeURIComponent(directory)}`)
      .then((r) => r.json())
      .then((data) => {
        const imgs: string[] = data.images ?? [];
        setImages(imgs);
        if (imgs.length > 0) setSelected(imgs[0]);
      });
  }, [directory]);

  const runDebug = useCallback(
    (filename: string) => {
      setSelected(filename);
      setDebugUrl(
        `/api/footage/debug-image?filename=${encodeURIComponent(filename)}&directory=${encodeURIComponent(directory)}&_t=${Date.now()}`,
      );
    },
    [directory],
  );

  // Auto-run debug on first selection
  useEffect(() => {
    if (selected && !debugUrl) runDebug(selected);
  }, [selected, debugUrl, runDebug]);

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      {/* Image list */}
      <div
        style={{
          width: "300px",
          flexShrink: 0,
          borderRight: "1px solid #ddd",
          overflowY: "auto",
          padding: "0.5rem",
        }}
      >
        {images.length === 0 && (
          <div style={{ color: "#999", fontSize: "0.85rem", padding: "0.5rem" }}>
            No images in Debug directory
          </div>
        )}
        {images.map((name) => (
          <div
            key={name}
            onClick={() => runDebug(name)}
            style={{
              padding: "0.4rem 0.6rem",
              marginBottom: "2px",
              cursor: "pointer",
              borderRadius: "4px",
              background: selected === name ? "#e8f0fe" : "transparent",
              fontSize: "0.875rem",
            }}
          >
            <div
              style={{
                fontWeight: selected === name ? 600 : 400,
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

      {/* Debug result */}
      <div style={{ flex: 1, background: "#1a1a1a", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
        {debugUrl ? (
          <img
            src={debugUrl}
            style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
            alt=""
          />
        ) : (
          <div style={{ color: "#999" }}>
            {images.length === 0 ? "No images" : "Select an image"}
          </div>
        )}
      </div>
    </div>
  );
}
