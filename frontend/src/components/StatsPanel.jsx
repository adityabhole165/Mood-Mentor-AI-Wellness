import { useEffect, useState } from "react";
import { getStats } from "../api/client.js";

const LABELS = {
  analysis_results: "Analyses run",
  recommendations: "Recommendations served",
  feedback: "Feedback events",
  emotion_history: "Emotion history rows",
};

export default function StatsPanel() {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setStats(await getStats());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <h2 style={{ fontSize: 18 }}>Database snapshot</h2>
        <button className="btn btn-ghost btn-sm" onClick={load} disabled={loading}>
          {loading ? <span className="spinner" /> : "Refresh"}
        </button>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      {stats ? (
        <div className="stat-grid">
          {Object.entries(stats).map(([key, value]) => (
            <div className="stat-card" key={key}>
              <div className="value">{value}</div>
              <div className="label">{LABELS[key] || key}</div>
            </div>
          ))}
        </div>
      ) : (
        !error && <p className="hint">Loading counts…</p>
      )}
    </div>
  );
}
