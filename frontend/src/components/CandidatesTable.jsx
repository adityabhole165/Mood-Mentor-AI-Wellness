const COLUMNS = [
  { key: "title", label: "Title" },
  { key: "rule_score", label: "Rule" },
  { key: "personalized_ml_score", label: "Personalized ML" },
  { key: "feedback_ml_score", label: "Feedback ML" },
  { key: "emotion_relevance", label: "Emotion rel." },
  { key: "content_similarity", label: "Semantic" },
  { key: "preference_score", label: "Preference" },
  { key: "collaborative_score", label: "Collaborative" },
  { key: "history_affinity", label: "History" },
  { key: "novelty", label: "Novelty" },
];

function fmt(v) {
  return typeof v === "number" ? v.toFixed(3) : v ?? "—";
}

export default function CandidatesTable({ candidates }) {
  if (!candidates?.length) return <p className="hint">No candidates were generated.</p>;

  const sorted = [...candidates].sort((a, b) => (b.rule_score ?? 0) - (a.rule_score ?? 0));

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
        <thead>
          <tr>
            {COLUMNS.map((c) => (
              <th
                key={c.key}
                style={{
                  textAlign: "left",
                  padding: "6px 10px",
                  borderBottom: "1px solid var(--border)",
                  color: "var(--ink-soft)",
                  whiteSpace: "nowrap",
                }}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.content_id}>
              {COLUMNS.map((c) => (
                <td
                  key={c.key}
                  style={{
                    padding: "6px 10px",
                    borderBottom: "1px solid var(--border)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {fmt(row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
