import { useEffect, useState } from "react";

interface SpeedingSection {
  avg_speed_kmh: number;
  location: string;
  date: string;
  start_lat: number;
  start_lon: number;
  video: string;
  second: number;
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

interface TopSpeedingProps {
  directory: string;
  onNavigateToVideo?: (filename: string, second: number) => void;
}

export default function TopSpeeding({ directory, onNavigateToVideo }: TopSpeedingProps) {
  const [speedingGroups, setSpeedingGroups] = useState<TopSpeedingGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/analytics/top-speeding-sections?directory=${encodeURIComponent(directory)}`)
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText);
        return r.json();
      })
      .then((data: TopSpeedingData) => setSpeedingGroups(data.groups))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [directory]);

  if (loading) return <div style={{ padding: "2rem", color: "#888" }}>Loading...</div>;
  if (error) return <div style={{ padding: "2rem", color: "#cc0000" }}>{error}</div>;

  return (
    <div>
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
                      <th style={thStyle}>Date</th>
                      <th style={thStyle}>Location</th>
                    </tr>
                  </thead>
                  <tbody>
                    {table.sections.map((section, idx) => (
                      <tr key={idx}>
                        <td style={tdStyle}>{idx + 1}</td>
                        <td style={tdStyle}>{section.avg_speed_kmh}</td>
                        <td style={tdStyle}>
                          <a
                            href="#"
                            onClick={(e) => {
                              e.preventDefault();
                              onNavigateToVideo?.(section.video, section.second);
                            }}
                            style={{ color: "#0066cc", textDecoration: "none" }}
                          >
                            {section.date}
                          </a>
                        </td>
                        <td style={tdStyle}>
                          <a
                            href={`https://www.google.com/maps?q=${section.start_lat},${section.start_lon}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={{ color: "#0066cc", textDecoration: "none" }}
                          >
                            {section.location}
                          </a>
                        </td>
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
