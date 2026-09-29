import EmotionResult from "./EmotionResult.jsx";

export default function ModelComparison({ comparison }) {
  const { bert, distilbert, agreeOnPrimary } = comparison;
  const severity = bert.emotional_state?.severity;

  return (
    <div>
      <div className="metric-row" style={{ marginBottom: 12 }}>
        <span className={agreeOnPrimary ? "pill pill-positive" : "pill pill-neutral"}>
          {agreeOnPrimary ? "BERT and DistilBERT agree on the primary emotion" : "Models disagree on the primary emotion"}
        </span>
      </div>

      {severity === "high" || severity === "critical" ? (
        <div className="error-banner" style={{ background: "var(--warm-soft)", color: "var(--warm)" }}>
          This reads as emotionally intense. The recommendations below are general wellness activities,
          not a substitute for professional support — if things feel like too much, consider reaching out
          to someone you trust or a mental-health professional.
        </div>
      ) : null}

      <div className="two-col">
        <EmotionResult result={bert} label="BERT" />
        <EmotionResult result={distilbert} label="DistilBERT" />
      </div>
    </div>
  );
}
