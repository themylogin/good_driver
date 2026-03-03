import { useEffect, useState } from "react";

interface SafetyIndexCumulativeProps {
  directory: string;
}

interface Bucket {
  speed_bucket: number;
  count: number;
}

export default function SafetyIndexCumulative({ directory }: SafetyIndexCumulativeProps) {
  const [noLeadAsSafe, setNoLeadAsSafe] = useState(false);
  const [buckets, setBuckets] = useState<Bucket[]>([]);

  useEffect(() => {
    fetch(
      `/api/analytics/safety-index-buckets?directory=${encodeURIComponent(directory)}&no_lead_as_safe=${noLeadAsSafe}`
    )
      .then((r) => (r.ok ? r.json() : { buckets: [] }))
      .then((data) => setBuckets(data.buckets));
  }, [directory, noLeadAsSafe]);

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
      {buckets.map((b) => (
        <div key={b.speed_bucket} style={{ marginBottom: "1rem" }}>
          <img
            src={`/api/analytics/safety-index-cumulative-chart?directory=${encodeURIComponent(directory)}&speed_bucket=${b.speed_bucket}&no_lead_as_safe=${noLeadAsSafe}`}
            alt={`Cumulative safety index at ${b.speed_bucket} km/h`}
            style={{ width: "100%", maxWidth: 1920 }}
          />
        </div>
      ))}
    </div>
  );
}
