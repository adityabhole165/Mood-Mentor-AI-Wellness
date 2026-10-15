import { useMemo } from "react";
import { EmotionTrendChart, EmotionFrequencyChart, EmotionScoreChart } from "./EmotionCharts.jsx";

// Graphs for a finished/ongoing .txt or .csv batch run, built from the rows
// analysed so far (rows still pending or failed are skipped).
export default function BatchCharts({ rows }) {
  const done = useMemo(() => (rows || []).filter((r) => r.status === "done" && r.result), [rows]);

  const trendRows = useMemo(
    () =>
      done.map((r) => ({
        period: `Row ${r.id}`,
        avg_intensity: r.result.emotional_state?.intensity,
        avg_polarity: r.result.emotional_state?.polarity_score,
        samples: 1,
      })),
    [done]
  );

  const frequencyRows = useMemo(() => {
    const counts = {};
    done.forEach((r) => {
      const e = r.result.emotional_state?.dominant_emotion || r.result.confidence?.primary_emotion;
      if (e) counts[e] = (counts[e] || 0) + 1;
    });
    return Object.entries(counts).map(([emotion, count]) => ({ emotion, count }));
  }, [done]);

  const scoreItems = useMemo(
    () =>
      done.map((r) => ({
        dominant_emotion: r.result.emotional_state?.dominant_emotion,
        intensity: r.result.emotional_state?.intensity,
        confidence: r.result.emotional_state?.emotion_confidence,
        polarity_score: r.result.emotional_state?.polarity_score,
      })),
    [done]
  );

  if (!done.length) return null;

  const titleStyle = { fontSize: 15, marginBottom: 8 };

  return (
    <div className="panel">
      <h2 style={{ fontSize: 16, marginBottom: 12 }}>Batch graphs ({done.length} analysed)</h2>

      <h3 className="section-title serif" style={titleStyle}>Emotion trend</h3>
      <EmotionTrendChart rows={trendRows} />

      <h3 className="section-title serif" style={{ ...titleStyle, marginTop: 18 }}>Emotion frequency</h3>
      <EmotionFrequencyChart rows={frequencyRows} />

      <h3 className="section-title serif" style={{ ...titleStyle, marginTop: 18 }}>Emotion scores</h3>
      <EmotionScoreChart items={scoreItems} />
    </div>
  );
}
