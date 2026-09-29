import { useState } from "react";
import AnalyzeForm from "./components/AnalyzeForm.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import EmotionResult from "./components/EmotionResult.jsx";
import RecommendationList from "./components/RecommendationList.jsx";
import CandidatesTable from "./components/CandidatesTable.jsx";
import BatchTable from "./components/BatchTable.jsx";
import FeedbackLog from "./components/FeedbackLog.jsx";
import StatsPanel from "./components/StatsPanel.jsx";
import HistoryTrends from "./components/HistoryTrends.jsx";
import { analyzeText, compareModels, ApiError } from "./api/client.js";

const TABS = [
  { id: "analyze", label: "Check in" },
  { id: "dashboard", label: "Dashboard" },
];

const BATCH_CONCURRENCY = 3;

async function runWithConcurrency(items, worker, concurrency, onItemDone) {
  const queue = [...items];
  async function next() {
    const item = queue.shift();
    if (!item) return;
    await worker(item).then(
      (result) => onItemDone(item, { status: "done", result }),
      (err) => onItemDone(item, { status: "error", error: err.message || "Failed" })
    );
    await next();
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, next));
}

export default function App() {
  const [tab, setTab] = useState("analyze");
  const [userId, setUserId] = useState("demo_user");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [singleResult, setSingleResult] = useState(null); // { comparison, primary }
  const [batchRows, setBatchRows] = useState(null); // [{ id, text, status, result?, error? }]
  const [selectedBatchId, setSelectedBatchId] = useState(null);
  const [feedbackLog, setFeedbackLog] = useState([]);

  function addFeedback(entry) {
    setFeedbackLog((log) => [...log, entry]);
  }

  async function handleSingleSubmit({ text, userId: uid, modelType, topK }) {
    setUserId(uid);
    setLoading(true);
    setError(null);
    setBatchRows(null);
    setSelectedBatchId(null);
    try {
      const comparison = await compareModels({ text, userId: uid, topK });
      setSingleResult({ comparison, primary: modelType.toLowerCase() });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong analyzing that entry.");
      setSingleResult(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleBatchSubmit({ rows, userId: uid, modelType, topK }) {
    setUserId(uid);
    setError(null);
    setSingleResult(null);
    setLoading(true);
    const initial = rows.map((r) => ({ ...r, status: "pending", result: null, error: null }));
    setBatchRows(initial);
    setSelectedBatchId(null);

    let firstDoneSelected = false;
    try {
      await runWithConcurrency(
        rows,
        (row) =>
          analyzeText({ text: row.text, userId: uid, modelType, topK, persistHistory: false }),
        BATCH_CONCURRENCY,
        (row, outcome) => {
          setBatchRows((prev) => prev.map((r) => (r.id === row.id ? { ...r, ...outcome } : r)));
          if (outcome.status === "done" && !firstDoneSelected) {
            firstDoneSelected = true;
            setSelectedBatchId(row.id);
          }
        }
      );
    } finally {
      setLoading(false);
    }
  }

  const selectedRow = batchRows?.find((r) => r.id === selectedBatchId && r.status === "done");
  const primaryResult = singleResult ? singleResult.comparison[singleResult.primary] : null;

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <h1>MoodMentor</h1>
          <p>A quiet feedback loop for how you're actually doing.</p>
        </div>
        <nav className="nav-tabs">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {tab === "analyze" ? (
        <div className="layout">
          <AnalyzeForm
            onSubmitSingle={handleSingleSubmit}
            onSubmitBatch={handleBatchSubmit}
            isLoading={loading}
            defaultUserId={userId}
          />

          <div className="feed">
            {error ? <div className="error-banner">{error}</div> : null}

            {!singleResult && !batchRows && !error ? (
              <div className="empty-state">
                <h2 className="serif">Nothing analyzed yet</h2>
                <p>
                  Write a journal entry, upload a .txt file, or bring a CSV of entries — the panel on
                  the left covers all three.
                </p>
              </div>
            ) : null}

            {singleResult ? (
              <>
                <ModelComparison comparison={singleResult.comparison} />
                <div>
                  <h2 style={{ fontSize: 16, marginBottom: 10 }}>
                    Recommended for you (from {singleResult.primary === "bert" ? "BERT" : "DistilBERT"})
                  </h2>
                  <RecommendationList
                    recommendations={primaryResult?.recommendations}
                    userId={userId}
                    onFeedback={addFeedback}
                  />
                </div>
                <div className="panel">
                  <h2 style={{ fontSize: 16, marginBottom: 10 }}>How the ranking was built</h2>
                  <CandidatesTable candidates={primaryResult?.candidates} />
                </div>
                <FeedbackLog entries={feedbackLog} />
              </>
            ) : null}

            {batchRows ? (
              <>
                <BatchTable rows={batchRows} selectedId={selectedBatchId} onSelect={setSelectedBatchId} />
                {selectedRow ? (
                  <>
                    <p className="hint">Details for row #{selectedRow.id}: “{selectedRow.text.slice(0, 120)}”</p>
                    <EmotionResult result={selectedRow.result} />
                    <div>
                      <h2 style={{ fontSize: 16, marginBottom: 10 }}>Recommended for this entry</h2>
                      <RecommendationList
                        recommendations={selectedRow.result.recommendations}
                        userId={userId}
                        onFeedback={addFeedback}
                      />
                    </div>
                    <FeedbackLog entries={feedbackLog} />
                  </>
                ) : null}
              </>
            ) : null}
          </div>
        </div>
      ) : (
        <div className="feed dashboard-wrap">
          <StatsPanel />
          <HistoryTrends userId={userId} />
        </div>
      )}
    </div>
  );
}