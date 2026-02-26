import { useEffect, useState } from "react";
import Calibrate from "./components/Calibrate";
import UploadFootage from "./components/UploadFootage";
import Analytics from "./components/Analytics";
import Settings from "./components/Settings";

export interface CameraParams {
  fx: number;
  pitch_degrees: number;
  camera_height_m: number;
  image_width: number;
  image_height: number;
}

type Tab = "settings" | "calibrate" | "upload" | "analytics";

const TABS: { id: Tab; label: string }[] = [
  { id: "settings", label: "Settings" },
  { id: "calibrate", label: "Calibrate" },
  { id: "upload", label: "Process footage" },
  { id: "analytics", label: "Analytics" },
];

type StartupState = "checking" | "model-missing" | "ready";

interface DownloadProgress {
  downloaded: number;
  total: number;
}

function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(0)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
}

export default function App() {
  const [startupState, setStartupState] = useState<StartupState>("checking");
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<"downloading" | "extracting" | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [modelName, setModelName] = useState("model");

  const [directory, setDirectory] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("settings");
  const [isOpening, setIsOpening] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);
  const [cameraParams, setCameraParams] = useState<CameraParams | null>(null);
  const [tabError, setTabError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const healthRes = await fetch("/api/health");
        const health = await healthRes.json();
        if (health.mode !== "desktop") {
          setStartupState("ready");
          return;
        }
        const statusRes = await fetch("/api/model/status");
        const status = await statusRes.json();
        if (status.name) setModelName(status.name);
        setStartupState(status.exists ? "ready" : "model-missing");
      } catch {
        // Health check failed (e.g. dev mode without backend); proceed normally.
        setStartupState("ready");
      }
    })();
  }, []);

  const handleDownload = async () => {
    setIsDownloading(true);
    setDownloadError(null);
    setDownloadProgress(null);
    setDownloadStatus("downloading");

    try {
      const response = await fetch("/api/model/download", { method: "POST" });
      if (!response.ok || !response.body) {
        throw new Error(`Request failed: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.error) {
              throw new Error(data.error);
            } else if (data.status === "extracting") {
              setDownloadStatus("extracting");
            } else if (data.status === "done") {
              setStartupState("ready");
              return;
            } else if (data.downloaded !== undefined) {
              setDownloadProgress({ downloaded: data.downloaded, total: data.total });
            }
          } catch (parseErr) {
            if (parseErr instanceof SyntaxError) continue;
            throw parseErr;
          }
        }
      }
    } catch (e) {
      setDownloadError(String(e));
    } finally {
      setIsDownloading(false);
      setDownloadStatus(null);
    }
  };

  // Startup check in progress
  if (startupState === "checking") {
    return (
      <div
        style={{
          fontFamily: "system-ui",
          height: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#888",
        }}
      >
        Loading...
      </div>
    );
  }

  // Model file missing — show download screen
  if (startupState === "model-missing") {
    const pct =
      downloadProgress && downloadProgress.total > 0
        ? Math.round((downloadProgress.downloaded / downloadProgress.total) * 100)
        : null;

    return (
      <div
        style={{
          fontFamily: "system-ui",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1rem",
          padding: "2rem",
        }}
      >
        <div style={{ fontSize: "1rem", color: "#333" }}>
          Model file <strong>{modelName}</strong> not found.
        </div>

        {!isDownloading && (
          <button
            onClick={handleDownload}
            style={{
              padding: "0.75rem 2rem",
              fontSize: "1rem",
              background: "#0066cc",
              color: "#fff",
              border: "none",
              borderRadius: "6px",
              cursor: "pointer",
            }}
          >
            Download model
          </button>
        )}

        {isDownloading && (
          <div style={{ width: "320px", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            <div style={{ fontSize: "0.9rem", color: "#555" }}>
              {downloadStatus === "extracting"
                ? "Extracting..."
                : pct !== null
                  ? `Downloading... ${pct}%`
                  : "Connecting..."}
            </div>
            <div
              style={{
                height: "8px",
                background: "#e0e0e0",
                borderRadius: "4px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: downloadStatus === "extracting" ? "100%" : pct !== null ? `${pct}%` : "0%",
                  background: "#0066cc",
                  borderRadius: "4px",
                  transition: "width 0.2s",
                }}
              />
            </div>
            {downloadProgress && downloadStatus !== "extracting" && (
              <div style={{ fontSize: "0.8rem", color: "#888", textAlign: "right" }}>
                {formatBytes(downloadProgress.downloaded)}
                {downloadProgress.total > 0 && ` / ${formatBytes(downloadProgress.total)}`}
              </div>
            )}
          </div>
        )}

        {downloadError && (
          <div style={{ color: "#cc0000", fontSize: "0.9rem", maxWidth: "360px", textAlign: "center" }}>
            {downloadError}
          </div>
        )}
      </div>
    );
  }

  // Normal app flow below

  const handleOpen = async () => {
    setIsOpening(true);
    setOpenError(null);
    try {
      const res = await fetch("/api/calibrate/open-directory", { method: "POST" });
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      setDirectory(data.directory);
    } catch (e) {
      setOpenError(String(e));
    } finally {
      setIsOpening(false);
    }
  };

  const handleTabClick = async (tabId: Tab) => {
    setTabError(null);
    if (tabId === "upload") {
      // Always re-fetch params so they update after the user recalibrates
      const res = await fetch(`/api/calibrate/params?directory=${encodeURIComponent(directory!)}`);
      if (!res.ok) {
        setTabError("Complete camera calibration first (need ≥ 2 fully annotated images).");
        return;
      }
      setCameraParams(await res.json());
    }
    setActiveTab(tabId);
  };

  // No directory open yet — show picker
  if (!directory) {
    return (
      <div
        style={{
          fontFamily: "system-ui",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1rem",
        }}
      >
        <button
          onClick={handleOpen}
          disabled={isOpening}
          style={{
            padding: "0.75rem 2rem",
            fontSize: "1rem",
            background: "#0066cc",
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            cursor: isOpening ? "default" : "pointer",
          }}
        >
          {isOpening ? "Opening..." : "Open directory"}
        </button>
        {openError && (
          <div style={{ color: "#cc0000", fontSize: "0.9rem" }}>{openError}</div>
        )}
      </div>
    );
  }

  // Directory open — show full app
  return (
    <div style={{ fontFamily: "system-ui", display: "flex", flexDirection: "column", height: "100vh" }}>
      <div style={{ display: "flex", borderBottom: "2px solid #ddd", flexShrink: 0 }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => handleTabClick(tab.id)}
            style={{
              padding: "0.75rem 1.5rem",
              border: "none",
              borderBottom: activeTab === tab.id ? "2px solid #0066cc" : "2px solid transparent",
              marginBottom: "-2px",
              background: "none",
              cursor: "pointer",
              fontFamily: "system-ui",
              fontSize: "1rem",
              color: activeTab === tab.id ? "#0066cc" : "#555",
              fontWeight: activeTab === tab.id ? 600 : 400,
            }}
          >
            {tab.label}
          </button>
        ))}
        <div style={{ marginLeft: "auto", padding: "0.5rem 1rem", fontSize: "0.8rem", color: "#999", alignSelf: "center" }}>
          {directory.split("/").pop()}
        </div>
      </div>

      {/* Tab error banner */}
      {tabError && (
        <div
          style={{
            padding: "0.4rem 1rem",
            background: "#fff8e1",
            borderBottom: "1px solid #ffe082",
            color: "#795548",
            fontSize: "0.85rem",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexShrink: 0,
          }}
        >
          {tabError}
          <button
            onClick={() => setTabError(null)}
            style={{ border: "none", background: "none", cursor: "pointer", color: "#795548", fontWeight: 700 }}
          >
            ✕
          </button>
        </div>
      )}

      <div style={{ flex: 1, overflow: "hidden" }}>
        {activeTab === "settings" && <Settings directory={directory} />}
        {activeTab === "calibrate" && <Calibrate directory={directory} />}
        {activeTab === "upload" && cameraParams && <UploadFootage directory={directory!} cameraParams={cameraParams} />}
        {activeTab === "analytics" && <Analytics />}
      </div>
    </div>
  );
}
