import { useEffect, useState } from "react";

interface SpeedLimit {
  speed_limit: number;
  count: number;
}

export default function Analytics({ directory }: { directory: string }) {
  const [limits, setLimits] = useState<SpeedLimit[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
    </div>
  );
}
