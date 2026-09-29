import { downloadCsv } from "../utils/csv.js";

function preview(text, n = 70) {
  return text.length > n ? `${text.slice(0, n)}…` : text;
}

export default function BatchTable({ rows, selectedId, onSelect }) {
  const done = rows.filter((r) => r.status === "done");
  const failed = rows.filter((r) => r.status === "error");

  const compounds = done.map((r) => r.result.sentiment.compound);
  const avgCompound = compounds.length ? compounds.reduce((a, b) => a + b, 0) / compounds.length : 0;
  const counts = { positive: 0, neutral: 0, negative: 0 };
  done.forEach((r) => counts[r.result.sentiment.label] !== undefined && counts[r.result.sentiment.label]++);

  function exportCsv() {
    const headers = ["id", "text", "sentiment_label", "compound", "primary_emotion", "confidence"];
    const data = done.map((r) => ({
      id: r.id,
      text: r.text,
      sentiment_label: r.result.sentiment.label,
      compound: r.result.sentiment.compound,
      primary_emotion: r.result.confidence?.primary_emotion,
      confidence: r.result.confidence?.primary_confidence,
    }));
    downloadCsv("mood_mentor_batch_report.csv", headers, data);
  }

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
        <h2 style={{ fontSize: 16 }}>Batch results ({done.length}/{rows.length})</h2>
        <button className="btn btn-ghost btn-sm" onClick={exportCsv} disabled={!done.length}>
          Download CSV report
        </button>
      </div>

      <div className="metric-row" style={{ marginBottom: 14 }}>
        <span className="pill">avg compound {avgCompound.toFixed(3)}</span>
        <span className="pill pill-positive">positive {counts.positive}</span>
        <span className="pill pill-neutral">neutral {counts.neutral}</span>
        <span className="pill pill-negative">negative {counts.negative}</span>
        {failed.length ? <span className="pill pill-negative">{failed.length} failed</span> : null}
      </div>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <thead>
            <tr>
              {["#", "Text", "Status", "Sentiment", "Compound", "Primary emotion", "Confidence", ""].map((h) => (
                <th
                  key={h}
                  style={{ textAlign: "left", padding: "6px 10px", borderBottom: "1px solid var(--border)", color: "var(--ink-soft)" }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} style={{ background: selectedId === r.id ? "var(--surface-sunken)" : "transparent" }}>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>{r.id}</td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>{preview(r.text)}</td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>
                  {r.status === "pending" ? <span className="spinner" /> : r.status === "error" ? "error" : "done"}
                </td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>
                  {r.result?.sentiment?.label ?? "—"}
                </td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>
                  {r.result ? Number(r.result.sentiment.compound).toFixed(2) : "—"}
                </td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", textTransform: "capitalize" }}>
                  {r.result?.confidence?.primary_emotion ?? "—"}
                </td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>
                  {r.result ? Number(r.result.confidence.primary_confidence).toFixed(2) : "—"}
                </td>
                <td style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)" }}>
                  {r.status === "done" ? (
                    <button className="btn btn-sm btn-ghost" onClick={() => onSelect(r.id)}>
                      View
                    </button>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
