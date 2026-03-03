import { useState } from "react";
import SafetyIndex from "./analytics/SafetyIndex";
import SafetyIndexCumulative from "./analytics/SafetyIndexCumulative";
import SpeedDistribution from "./analytics/SpeedDistribution";
import SpeedDistributionCumulative from "./analytics/SpeedDistributionCumulative";
import TopSpeeding from "./analytics/TopSpeeding";

type AnalyticsTab = "speed-distribution" | "speed-distribution-cumulative" | "top-speeding" | "safety-index" | "safety-index-cumulative";

const TABS: { id: AnalyticsTab; label: string }[] = [
  { id: "speed-distribution", label: "Speed Distribution" },
  { id: "speed-distribution-cumulative", label: "Speed Distribution (Cumulative)" },
  { id: "top-speeding", label: "Top Speeding" },
  { id: "safety-index", label: "Safety Index" },
  { id: "safety-index-cumulative", label: "Safety Index (Cumulative)" },
];

interface AnalyticsProps {
  directory: string;
  onNavigateToVideo?: (filename: string, second: number) => void;
}

export default function Analytics({ directory, onNavigateToVideo }: AnalyticsProps) {
  const [activeTab, setActiveTab] = useState<AnalyticsTab>("speed-distribution");

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", borderBottom: "1px solid #ddd", flexShrink: 0 }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "0.5rem 1rem",
              border: "none",
              borderBottom: activeTab === tab.id ? "2px solid #0066cc" : "2px solid transparent",
              marginBottom: "-1px",
              background: "none",
              cursor: "pointer",
              fontFamily: "system-ui",
              fontSize: "0.85rem",
              color: activeTab === tab.id ? "#0066cc" : "#555",
              fontWeight: activeTab === tab.id ? 600 : 400,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "1rem" }}>
        {activeTab === "speed-distribution" && <SpeedDistribution directory={directory} />}
        {activeTab === "speed-distribution-cumulative" && <SpeedDistributionCumulative directory={directory} />}
        {activeTab === "top-speeding" && <TopSpeeding directory={directory} onNavigateToVideo={onNavigateToVideo} />}
        {activeTab === "safety-index" && <SafetyIndex directory={directory} />}
        {activeTab === "safety-index-cumulative" && <SafetyIndexCumulative directory={directory} />}
      </div>
    </div>
  );
}
