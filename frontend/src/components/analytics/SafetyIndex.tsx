import { useState } from "react";

interface SafetyIndexProps {
  directory: string;
}

export default function SafetyIndex({ directory }: SafetyIndexProps) {
  const [noLeadAsSafe, setNoLeadAsSafe] = useState(false);

  return (
    <div>
      <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", margin: "0 0 1rem", cursor: "pointer" }}>
        <input
          type="checkbox"
          checked={noLeadAsSafe}
          onChange={(e) => setNoLeadAsSafe(e.target.checked)}
        />
        <span style={{ fontSize: "0.9rem" }}>Assume safety index of 1.0 when not following any car</span>
      </label>
      <div style={{
        background: "#1a1a2e",
        border: "1px solid #333",
        borderRadius: 8,
        padding: "1rem 1.25rem",
        marginBottom: "1rem",
        maxWidth: 800,
      }}>
        <h4 style={{ margin: "0 0 0.5rem", fontSize: "0.95rem", color: "#e0e0e0" }}>
          How to read the chart
        </h4>
        <ul style={{ margin: 0, paddingLeft: "1.25rem", color: "#b0b0b0", fontSize: "0.85rem", lineHeight: 1.7 }}>
          <li><strong style={{ color: "#e0e0e0" }}>Median</strong> (bars) — half the time your safety index is above this, half below</li>
          <li><strong style={{ color: "#e0e0e0" }}>75th percentile</strong> — you are safer than this 25% of the time (your best driving)</li>
          <li><strong style={{ color: "#e0e0e0" }}>25th percentile</strong> — you are less safe than this 25% of the time (your worst driving)</li>
          <li>The gap between p25 and p75 shows how consistent your following distance is: tight = consistent, wide = lots of variation</li>
        </ul>
      </div>
      <img
        src={`/api/analytics/safety-index-chart?directory=${encodeURIComponent(directory)}&no_lead_as_safe=${noLeadAsSafe}`}
        alt="Safety index by driving speed"
        style={{ width: "100%", maxWidth: 1920 }}
      />
    </div>
  );
}
