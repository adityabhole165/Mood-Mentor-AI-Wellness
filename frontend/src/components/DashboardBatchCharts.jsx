import { useEffect, useMemo, useState } from "react";
import { loadBatchEntries, clearBatchEntries } from "../utils/batchStore.js";
import { EmotionTrendChart, EmotionFrequencyChart, EmotionScoreChart } from "./EmotionCharts.jsx";

function dayKey(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function avg(list) {
  const nums = list.map(Number).filter(Number.isFinite);
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
}

// Dashboard section: graphs of every statement analysed from .txt / .csv files.
export default function DashboardBatchCharts({ userId }) {
  const [entries, setEntries] = useState(() => loadBatchEntries(userId));
  const [confirmClear, setConfirmClear] = useState(false);

  useEffect(() => {
    const refresh = () => setEntries(loadBatchEntries(userId));
    refresh();
    window.addEventListener("mm-batch-updated", refresh);
    window.addEventListener("storage", refresh);
    return () => {
      window.removeEventListener("mm-batch-updated", refresh);
      window.removeEventListener("storage", refresh);
    };
  }, [userId]);

  const trendRows = useMemo(() => {
    const byDay = new Map();
    entries.forEach((e) => {
      const k = dayKey(e.timestamp);
      if (!k) return;
      if (!byDay.has(k)) byDay.set(k, []);
      byDay.get(k).push(e);
    });
    return [...byDay.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([period, list]) => ({
        period,
        avg_intensity: avg(list.map((e) => e.intensity)),
        avg_polarity: avg(list.map((e) => e.polarity_score)),
        samples: list.length,
      }));
  }, [entries]);

  const frequencyRows = useMemo(() => {
    const counts = {};
    entries.forEach((e) => {
      if (e.dominant_emotion) counts[e.dominant_emotion] = (counts[e.dominant_emotion] || 0) + 1;
    });
    return Object.entries(counts).map(([emotion, count]) => ({ emotion, count }));
  }, [entries]);

  const titleStyle = { fontSize: 15, marginBottom: 8 };
  const files = new Set(entries.map((e) => e.source).filter(Boolean));

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <h2 style={{ fontSize: 18 }}>Analysed .txt / .csv data</h2>
        {entries.length ? (
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              if (!confirmClear) return setConfirmClear(true);
              clearBatchEntries(userId);
              setConfirmClear(false);
            }}
          >
            {confirmClear ? "Confirm clear" : "Clear this data"}
          </button>
        ) : null}
      </div>

      {!entries.length ? (
        <p className="hint" style={{ marginTop: 10 }}>
          Nothing yet. Analyze a .txt or .csv file on the Analyze page and every evaluated statement will be graphed here.
        </p>
      ) : (
        <>
          <p className="hint" style={{ margin: "6px 0 14px" }}>
            {entries.length} statement(s) analysed{files.size ? ` from ${[...files].join(" / ")} uploads` : ""}.
          </p>

          <h3 className="section-title serif" style={titleStyle}>Emotion trend</h3>
          <EmotionTrendChart rows={trendRows} />

          <h3 className="section-title serif" style={{ ...titleStyle, marginTop: 18 }}>Emotion frequency</h3>
          <EmotionFrequencyChart rows={frequencyRows} />

          <h3 className="section-title serif" style={{ ...titleStyle, marginTop: 18 }}>Emotion scores</h3>
          <EmotionScoreChart items={entries} />
        </>
      )}
    </div>
  );
}
