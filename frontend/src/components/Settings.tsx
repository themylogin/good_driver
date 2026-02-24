import { useState, useEffect } from "react";

interface AppSettings {
  osrm_url: string;
  nominatim_url: string;
}

type CheckStatus = "idle" | "checking" | "ok" | "error";

interface CheckState {
  status: CheckStatus;
  message?: string;
}

const FIELD_STYLE: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "0.4rem",
  marginBottom: "1.5rem",
};

const LABEL_STYLE: React.CSSProperties = {
  fontWeight: 600,
  fontSize: "0.95rem",
};

const DESC_STYLE: React.CSSProperties = {
  fontSize: "0.82rem",
  color: "#666",
  margin: 0,
};

const INPUT_ROW_STYLE: React.CSSProperties = {
  display: "flex",
  gap: "0.5rem",
  alignItems: "center",
};

const INPUT_STYLE: React.CSSProperties = {
  fontFamily: "monospace",
  fontSize: "0.9rem",
  padding: "0.4rem 0.6rem",
  border: "1px solid #ccc",
  borderRadius: "4px",
  width: "340px",
};

function checkColor(status: CheckStatus): string {
  if (status === "ok") return "#2e7d32";
  if (status === "error") return "#c62828";
  return "#555";
}

export default function Settings({ directory }: { directory: string }) {
  const [settings, setSettings] = useState<AppSettings>({
    osrm_url: "http://localhost:5000",
    nominatim_url: "http://localhost:8080",
  });
  const [osrmCheck, setOsrmCheck] = useState<CheckState>({ status: "idle" });
  const [nominatimCheck, setNominatimCheck] = useState<CheckState>({ status: "idle" });

  useEffect(() => {
    fetch(`/api/settings?directory=${encodeURIComponent(directory)}`)
      .then((r) => r.json())
      .then(setSettings)
      .catch(() => {});
  }, [directory]);

  const save = async (updated: AppSettings) => {
    await fetch(`/api/settings?directory=${encodeURIComponent(directory)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updated),
    });
  };

  const handleChange = (field: keyof AppSettings, value: string) => {
    const updated = { ...settings, [field]: value };
    setSettings(updated);
    // reset check status when URL changes
    if (field === "osrm_url") setOsrmCheck({ status: "idle" });
    if (field === "nominatim_url") setNominatimCheck({ status: "idle" });
  };

  const handleBlur = () => {
    save(settings);
  };

  const check = async (service: "osrm" | "nominatim") => {
    const url = service === "osrm" ? settings.osrm_url : settings.nominatim_url;
    const setCheck = service === "osrm" ? setOsrmCheck : setNominatimCheck;
    setCheck({ status: "checking" });
    try {
      const res = await fetch("/api/settings/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, service }),
      });
      const data = await res.json();
      setCheck({ status: data.ok ? "ok" : "error", message: data.message });
    } catch {
      setCheck({ status: "error", message: "Request failed" });
    }
  };

  const renderCheckResult = (state: CheckState) => {
    if (state.status === "idle") return null;
    if (state.status === "checking") {
      return <span style={{ fontSize: "0.85rem", color: "#555" }}>Checking…</span>;
    }
    return (
      <span style={{ fontSize: "0.85rem", color: checkColor(state.status), whiteSpace: "nowrap" }}>
        {state.status === "ok" ? "✓" : "✗"} {state.message}
      </span>
    );
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "600px" }}>
      <div style={{ marginBottom: "2rem" }}>
        <h3 style={{ margin: "0 0 1.25rem", fontSize: "0.9rem", textTransform: "uppercase", letterSpacing: "0.06em", color: "#888" }}>
          External services
        </h3>
        <p style={{ ...DESC_STYLE, marginBottom: "1.5rem" }}>
          See <strong>README.md</strong> for setup instructions.
        </p>

        <div style={FIELD_STYLE}>
          <label style={LABEL_STYLE}>OSRM API URL</label>
          <p style={DESC_STYLE}>
            Matches your recorded GPS tracks to actual roads and retrieves speed limits for each road segment.
          </p>
          <div style={INPUT_ROW_STYLE}>
            <input
              style={INPUT_STYLE}
              type="text"
              value={settings.osrm_url}
              onChange={(e) => handleChange("osrm_url", e.target.value)}
              onBlur={handleBlur}
              placeholder="http://localhost:5000"
            />
            <button
              onClick={() => check("osrm")}
              disabled={osrmCheck.status === "checking"}
              style={{
                padding: "0.4rem 0.9rem",
                fontSize: "0.88rem",
                border: "1px solid #bbb",
                borderRadius: "4px",
                background: "#f5f5f5",
                cursor: osrmCheck.status === "checking" ? "default" : "pointer",
              }}
            >
              Check
            </button>
            {renderCheckResult(osrmCheck)}
          </div>
        </div>

        <div style={FIELD_STYLE}>
          <label style={LABEL_STYLE}>Nominatim API URL</label>
          <p style={DESC_STYLE}>
            Turns GPS coordinates into readable location names — street names, towns, and landmarks shown in the analytics.
          </p>
          <div style={INPUT_ROW_STYLE}>
            <input
              style={INPUT_STYLE}
              type="text"
              value={settings.nominatim_url}
              onChange={(e) => handleChange("nominatim_url", e.target.value)}
              onBlur={handleBlur}
              placeholder="http://localhost:8080"
            />
            <button
              onClick={() => check("nominatim")}
              disabled={nominatimCheck.status === "checking"}
              style={{
                padding: "0.4rem 0.9rem",
                fontSize: "0.88rem",
                border: "1px solid #bbb",
                borderRadius: "4px",
                background: "#f5f5f5",
                cursor: nominatimCheck.status === "checking" ? "default" : "pointer",
              }}
            >
              Check
            </button>
            {renderCheckResult(nominatimCheck)}
          </div>
        </div>
      </div>
    </div>
  );
}
