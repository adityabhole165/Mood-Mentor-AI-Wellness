function polarityPillClass(polarity) {
  if (polarity === "positive") return "pill pill-positive";
  if (polarity === "negative") return "pill pill-negative";
  return "pill pill-neutral";
}

export default function EmotionResult({ result, label }) {
  const state = result.emotional_state || {};
  const sentiment = result.sentiment || {};
  const trend = result.emotion_history?.trend;

  const probabilities = Object.entries(state.emotion_probabilities || {}).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <div className="panel">
      {label ? <p className="hint" style={{ marginBottom: 6 }}>{label}</p> : null}
      <div className="state-hero">
        <div className="dominant serif">
          <small>Dominant emotion</small>
          {state.dominant_emotion || "—"}
        </div>
        <div className="metric-row" style={{ marginTop: 0 }}>
          <span className={polarityPillClass(state.polarity)}>
            {state.polarity} · {Number(state.polarity_score ?? 0).toFixed(2)}
          </span>
          <span className="pill">severity: {state.severity || "—"}</span>
          {state.mixed_state ? <span className="pill">mixed emotional state</span> : null}
        </div>
      </div>

      <div className="metric-row">
        <span className="pill">intensity {Math.round((state.intensity ?? 0) * 100)}%</span>
        <span className="pill">confidence {Math.round((state.emotion_confidence ?? 0) * 100)}%</span>
        <span className="pill">uncertainty {Math.round((state.uncertainty ?? 0) * 100)}%</span>
        <span className="pill">
          sentiment {sentiment.label} ({Number(sentiment.compound ?? 0).toFixed(2)})
        </span>
        <span className="pill">model: {result.emotion_model}</span>
      </div>

      <div className="metric-row">
        <span className="pill pill-positive">
          positive load {Number(state.positive_emotion_load ?? 0).toFixed(2)}
        </span>
        <span className="pill pill-negative">
          negative load {Number(state.negative_emotion_load ?? 0).toFixed(2)}
        </span>
      </div>

      <div className="bars">
        {probabilities.map(([emotion, score]) => (
          <div className="bar-row" key={emotion}>
            <span className="label">{emotion}</span>
            <span className="bar-track">
              <span className="bar-fill" style={{ width: `${Math.round(score * 100)}%` }} />
            </span>
            <span className="value">{Math.round(score * 100)}%</span>
          </div>
        ))}
      </div>

      <p className="hint" style={{ marginTop: 10, marginBottom: 0 }}>
        Triggered (≥ {Math.round((result.confidence?.threshold ?? 0.5) * 100)}%):{" "}
        {state.triggered_emotions?.length ? state.triggered_emotions.join(", ") : "—"}
      </p>

      {trend ? (
        <>
          <hr className="divider" />
          <p className="hint" style={{ margin: 0 }}>
            Trend: {trend.polarity_trend || "steady"}
            {trend.repeated_emotions?.length
              ? ` · recurring: ${trend.repeated_emotions.join(", ")}`
              : ""}
          </p>
        </>
      ) : null}
    </div>
  );
}
