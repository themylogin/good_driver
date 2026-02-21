import { useState } from "react";
import Calibrate from "./components/Calibrate";
import UploadFootage from "./components/UploadFootage";
import Analytics from "./components/Analytics";

type Tab = "calibrate" | "upload" | "analytics";

const TABS: { id: Tab; label: string }[] = [
  { id: "calibrate", label: "Calibrate" },
  { id: "upload", label: "Upload footage" },
  { id: "analytics", label: "Analytics" },
];

export default function App() {
  const [directory, setDirectory] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("calibrate");
  const [isOpening, setIsOpening] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);

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
            onClick={() => setActiveTab(tab.id)}
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

      <div style={{ flex: 1, overflow: "hidden" }}>
        {activeTab === "calibrate" && <Calibrate directory={directory} />}
        {activeTab === "upload" && <UploadFootage />}
        {activeTab === "analytics" && <Analytics />}
      </div>
    </div>
  );
}
