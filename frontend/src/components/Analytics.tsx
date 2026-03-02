import { useEffect, useState } from "react";

interface SpeedLimit {
  speed_limit: number;
  count: number;
}

interface SpeedingSection {
  avg_speed_kmh: number;
  location: string;
}

interface SpeedingTable {
  window_seconds: number;
  window_label: string;
  sections: SpeedingSection[];
}

interface TopSpeedingGroup {
  speed_limit: number;
  tables: SpeedingTable[];
}

interface TopSpeedingData {
  groups: TopSpeedingGroup[];
}

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "0.5rem 1rem",
  borderBottom: "2px solid #ddd",
  fontWeight: 600,
};

const tdStyle: React.CSSProperties = {
  padding: "0.4rem 1rem",
  borderBottom: "1px solid #eee",
};

export default function Analytics({ directory }: { directory: string }) {
  const [limits, setLimits] = useState<SpeedLimit[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [speedingGroups, setSpeedingGroups] = useState<TopSpeedingGroup[]>([]);
  const [speedingLoading, setSpeedingLoading] = useState(true);
  const [speedingError, setSpeedingError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/analytics/speed-distribution?directory=${encodeURIComponent(directory)}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText);
        return r.json();
      })
      .then((data) => setLimits(data.limits))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [directory]);

  useEffect(() => {
    setSpeedingLoading(true);
    setSpeedingError(null);
    fetch(`/api/analytics/top-speeding-sections?directory=${encodeURIComponent(directory)}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText);
        return r.json();
      })
      .then((data: TopSpeedingData) => setSpeedingGroups(data.groups))
      .catch((e) => setSpeedingError(String(e)))
      .finally(() => setSpeedingLoading(false));
  }, [directory]);

  if (loading) return <div style={{ padding: "2rem", color: "#888" }}>Loading...</div>;
  if (error) return <div style={{ padding: "2rem", color: "#cc0000" }}>{error}</div>;
  if (limits.length === 0)
    return <div style={{ padding: "2rem", color: "#888" }}>No speed distribution data available. Process some videos with GPS and snap-to-road first.</div>;

  return (
    <div style={{ padding: "1rem", overflowY: "auto", height: "100%" }}>
      {limits.map((l) => (
        <div key={l.speed_limit} style={{ marginBottom: "1rem" }}>
          <img
            src={`/api/analytics/speed-distribution-chart?directory=${encodeURIComponent(directory)}&speed_limit=${l.speed_limit}`}
            alt={`Speed distribution for ${l.speed_limit} km/h limit`}
            style={{ width: "100%", maxWidth: 1920 }}
          />
        </div>
      ))}

      {speedingLoading && (
        <div style={{ padding: "1rem", color: "#888" }}>Loading speeding sections...</div>
      )}
      {speedingError && (
        <div style={{ padding: "1rem", color: "#cc0000" }}>{speedingError}</div>
      )}
      {speedingGroups.map((group) => (
        <div key={group.speed_limit} style={{ padding: "1rem 0" }}>
          <h2 style={{ marginBottom: "1rem" }}>
            Top speeding sections (speed limit: {group.speed_limit} km/h)
          </h2>
          {group.tables.map((table) => (
            <div key={table.window_seconds} style={{ marginBottom: "2rem" }}>
              <h3 style={{ marginBottom: "0.5rem" }}>{table.window_label}</h3>
              {table.sections.length === 0 ? (
                <div style={{ color: "#888" }}>No data for this window duration.</div>
              ) : (
                <table style={{ borderCollapse: "collapse", width: "100%" }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>#</th>
                      <th style={thStyle}>Avg speed (km/h)</th>
                      <th style={thStyle}>Location</th>
                    </tr>
                  </thead>
                  <tbody>
                    {table.sections.map((section, idx) => (
                      <tr key={idx}>
                        <td style={tdStyle}>{idx + 1}</td>
                        <td style={tdStyle}>{section.avg_speed_kmh}</td>
                        <td style={tdStyle}>{section.location}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
