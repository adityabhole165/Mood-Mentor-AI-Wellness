import { useState, useEffect } from "react";
import {
  getEmotionHistory,
  getFeedbackHistory,
  getRecommendationHistory,
  getTrends,
  getReportUrl,
  deleteUserData,
} from "../api/client.js";

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 22 }}>
      <h3 className="section-title serif" style={{ fontSize: 15 }}>
        {title}
      </h3>
      {children}
    </div>
  );
}

const tableWrapStyle = { overflowX: "auto", border: "1px solid var(--border, #e5e5e5)", borderRadius: 8 };
const tableStyle = { width: "100%", borderCollapse: "collapse", fontSize: 13 };
const thStyle = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "1px solid var(--border, #e5e5e5)",
  background: "var(--surface-muted, #fafafa)",
  whiteSpace: "nowrap",
  fontWeight: 600,
};
const tdStyle = {
  padding: "6px 10px",
  borderBottom: "1px solid var(--border, #f0f0f0)",
  whiteSpace: "nowrap",
  maxWidth: 320,
  overflow: "hidden",
  textOverflow: "ellipsis",
};

function formatCell(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(4);
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}T/.test(value)) {
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
  }
  return String(value);
}

const paginationBarStyle = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: 10,
  marginTop: 8,
  flexWrap: "wrap",
};
const pageInfoStyle = { fontSize: 12, color: "var(--ink-soft, #666)" };

const PAGE_SIZE = 10;

// rows: full (already filtered) row set. When `paginate` is true, only
// PAGE_SIZE rows are rendered at a time, with Prev/Next controls below
// the table. The page resets to 1 whenever the underlying row set changes
// (new data loaded, or the search query narrows/widens the results).
function DataTable({ rows, columns, emptyMessage, paginate = false }) {
  const [page, setPage] = useState(1);
  const rowsSignature = paginate ? JSON.stringify(rows) : null;

  useEffect(() => {
    if (paginate) setPage(1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rowsSignature]);

  if (!rows || rows.length === 0) {
    return <p className="hint">{emptyMessage || "No records yet."}</p>;
  }

  const totalPages = paginate ? Math.max(1, Math.ceil(rows.length / PAGE_SIZE)) : 1;
  const currentPage = Math.min(page, totalPages);
  const visibleRows = paginate
    ? rows.slice((currentPage - 1) * PAGE_SIZE, (currentPage - 1) * PAGE_SIZE + PAGE_SIZE)
    : rows;
  const rangeStart = paginate ? (currentPage - 1) * PAGE_SIZE + 1 : 1;
  const rangeEnd = paginate ? Math.min(currentPage * PAGE_SIZE, rows.length) : rows.length;

  return (
    <div>
      <div style={tableWrapStyle}>
        <table style={tableStyle}>
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key} style={thStyle}>
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td key={col.key} style={tdStyle} title={formatCell(row[col.key])}>
                    {formatCell(row[col.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {paginate && rows.length > PAGE_SIZE ? (
        <div style={paginationBarStyle}>
          <span style={pageInfoStyle}>
            Showing {rangeStart}–{rangeEnd} of {rows.length}
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
            >
              Prev
            </button>
            <span style={pageInfoStyle}>
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
            >
              Next
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

const HISTORY_COLUMNS = [
  { key: "timestamp", label: "Timestamp" },
  { key: "dominant_emotion", label: "Emotion" },
  { key: "intensity", label: "Intensity" },
  { key: "polarity", label: "Polarity" },
  { key: "polarity_score", label: "Polarity score" },
  { key: "confidence", label: "Confidence" },
  { key: "triggered_emotions", label: "Triggered" },
];

const RECOMMENDATION_COLUMNS = [
  { key: "rank", label: "Rank" },
  { key: "title", label: "Title" },
  { key: "content_id", label: "Content ID" },
  { key: "score", label: "Score" },
  { key: "created_at", label: "Created" },
];

const FEEDBACK_COLUMNS = [
  { key: "timestamp", label: "Timestamp" },
  { key: "event", label: "Event" },
  { key: "content_id", label: "Content ID" },
  { key: "rating", label: "Rating" },
];

const TREND_COLUMNS = [
  { key: "period", label: "Period" },
  { key: "avg_intensity", label: "Avg intensity" },
  { key: "avg_polarity", label: "Avg polarity" },
  { key: "samples", label: "Samples" },
];

const FREQUENCY_COLUMNS = [
  { key: "emotion", label: "Emotion" },
  { key: "count", label: "Count" },
];

function filterRows(rows, query) {
  const q = query.trim().toLowerCase();
  if (!q) return rows;
  return rows.filter((row) => JSON.stringify(row).toLowerCase().includes(q));
}

export default function HistoryTrends({ userId }) {
  const [granularity, setGranularity] = useState("daily");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState({
    emotions: null,
    recommendations: null,
    feedback: null,
    trends: null,
  });
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [deleteNote, setDeleteNote] = useState("");
  const [query, setQuery] = useState("");

  async function loadAll() {
    setLoading(true);
    setError(null);
    try {
      const [emotions, recommendations, feedback, trends] = await Promise.all([
        getEmotionHistory(userId),
        getRecommendationHistory(userId),
        getFeedbackHistory(userId),
        getTrends(userId, granularity),
      ]);
      setData({ emotions, recommendations, feedback, trends });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // Auto-load on mount and whenever the active user or granularity changes.
  useEffect(() => {
    loadAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, granularity]);

  async function handleDelete() {
    if (!deleteConfirm) {
      setDeleteConfirm(true);
      return;
    }
    try {
      await deleteUserData(userId);
      setDeleteNote(`All records for "${userId}" were deleted.`);
      setData({ emotions: null, recommendations: null, feedback: null, trends: null });
    } catch (err) {
      setDeleteNote(err.message);
    } finally {
      setDeleteConfirm(false);
    }
  }

  function renderSection(section, columns, emptyMessage, paginate = false) {
    if (!section) return <p className="hint">Not loaded yet.</p>;
    if (!section.available) return <p className="not-available">Not available on this backend.</p>;
    const rows = Array.isArray(section.data?.items)
      ? section.data.items
      : Array.isArray(section.data)
      ? section.data
      : [];
    const filtered = filterRows(rows, query);
    return <DataTable rows={filtered} columns={columns} emptyMessage={emptyMessage} paginate={paginate} />;
  }

  const trendSection = data.trends;
  const trendRows =
    trendSection?.available && trendSection.data ? filterRows(trendSection.data.trend || [], query) : [];
  const frequencyRows =
    trendSection?.available && trendSection.data
      ? filterRows(trendSection.data.emotion_frequency || [], query)
      : [];

  return (
    <div className="panel">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <h2 style={{ fontSize: 18 }}>History &amp; trends</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <select value={granularity} onChange={(e) => setGranularity(e.target.value)}>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
          <button className="btn btn-ghost btn-sm" onClick={loadAll} disabled={loading}>
            {loading ? <span className="spinner" /> : `Reload for ${userId}`}
          </button>
        </div>
      </div>

      {error ? <div className="error-banner" style={{ marginTop: 14 }}>{error}</div> : null}

      <div className="field" style={{ marginTop: 14, marginBottom: 0 }}>
        <label htmlFor="historySearch">Search history / recommendations / feedback</label>
        <input
          id="historySearch"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="emotion, content id, event…"
        />
      </div>

      <hr className="divider" />

      <Section title="Emotional trend">
        {!trendSection ? (
          <p className="hint">Not loaded yet.</p>
        ) : !trendSection.available ? (
          <p className="not-available">Not available on this backend.</p>
        ) : (
          <>
            <DataTable rows={trendRows} columns={TREND_COLUMNS} emptyMessage="No trend data yet." />
            <div style={{ marginTop: 10 }}>
              <p className="hint" style={{ marginBottom: 6 }}>Emotion frequency</p>
              <DataTable rows={frequencyRows} columns={FREQUENCY_COLUMNS} emptyMessage="No emotion frequency data yet." />
            </div>
          </>
        )}
      </Section>

      <Section title="Recent emotion history">
        {renderSection(data.emotions, HISTORY_COLUMNS, "No emotion history yet.", true)}
      </Section>

      <Section title="Recent recommendations">
        {renderSection(data.recommendations, RECOMMENDATION_COLUMNS, "No recommendations yet.", true)}
      </Section>

      <Section title="Recent feedback">
        {renderSection(data.feedback, FEEDBACK_COLUMNS, "No feedback yet.", true)}
      </Section>

      <hr className="divider" />

      <Section title="Export">
        <div style={{ display: "flex", gap: 8 }}>
          <a className="btn btn-ghost btn-sm" href={getReportUrl(userId, "csv")}>
            Download CSV report
          </a>
          <a className="btn btn-ghost btn-sm" href={getReportUrl(userId, "json")}>
            Download JSON report
          </a>
        </div>
      </Section>

      <Section title="Privacy">
        <button className="btn btn-sm btn-reject" onClick={handleDelete}>
          {deleteConfirm ? `Confirm delete all data for "${userId}"` : "Delete my data"}
        </button>
        {deleteNote ? <p className="feedback-note" style={{ color: "var(--ink-soft)" }}>{deleteNote}</p> : null}
      </Section>
    </div>
  );
}