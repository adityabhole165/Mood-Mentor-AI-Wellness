import { useEffect, useRef, useState } from "react";
import AnalyzeForm from "./components/AnalyzeForm.jsx";
import ModelComparison from "./components/ModelComparison.jsx";
import EmotionResult from "./components/EmotionResult.jsx";
import RecommendationList from "./components/RecommendationList.jsx";
import CandidatesTable from "./components/CandidatesTable.jsx";
import BatchTable from "./components/BatchTable.jsx";
import FeedbackLog from "./components/FeedbackLog.jsx";
import StatsPanel from "./components/StatsPanel.jsx";
import HistoryTrends from "./components/HistoryTrends.jsx";
import SiteHeader from "./components/SiteHeader.jsx";
import SiteFooter from "./components/SiteFooter.jsx";
import ApiStatusBanner from "./components/ApiStatusBanner.jsx";
import CrisisBanner, { shouldShowCrisis } from "./components/CrisisBanner.jsx";
import ConsentCard from "./components/ConsentCard.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import AuraPanel from "./components/AuraPanel.jsx";
import WordHighlight from "./components/WordHighlight.jsx";
import GuidedReset from "./components/GuidedReset.jsx";
import YearInPixels from "./components/YearInPixels.jsx";
import Landing from "./pages/Landing.jsx";
import AuthPage from "./pages/AuthPage.jsx";
import NotFound from "./pages/NotFound.jsx";
import { About, Privacy, Terms } from "./pages/LegalPages.jsx";
import { useAuth } from "./auth/AuthContext.jsx";
import useHashRoute from "./hooks/useHashRoute.js";
import useTheme from "./hooks/useTheme.js";
import { analyzeText, compareModels, ApiError } from "./api/client.js";

const BATCH_CONCURRENCY = 3;
const MAX_CHARS = 5000;
const CONSENT_KEY = "moodmentor.consent.v1";

const PAGE_TITLES = {
  "/": "MoodMentor — understand how you feel",
  "/app": "Check in · MoodMentor",
  "/dashboard": "Dashboard · MoodMentor",
  "/about": "About · MoodMentor",
  "/privacy": "Privacy Policy · MoodMentor",
  "/terms": "Terms of Use · MoodMentor",
  "/login": "Sign in · MoodMentor",
  "/signup": "Create account · MoodMentor",
};

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

function readConsent() {
  try {
    return localStorage.getItem(CONSENT_KEY) === "yes";
  } catch {
    return false;
  }
}

// Crisis resources + aura + word-level explanation for one analysis result.
function Insights({ result }) {
  const state = result?.emotional_state;
  return (
    <>
      {shouldShowCrisis(state) ? <CrisisBanner /> : null}
      <AuraPanel result={result} />
      <WordHighlight
        key={`${result.input_text}|${result.emotion_model}`}
        text={result.input_text}
        modelType={result.emotion_model}
        target={state?.dominant_emotion}
      />
    </>
  );
}

export default function App() {
  const [route, navigate] = useHashRoute();
  const [theme, toggleTheme] = useTheme();
  const { userId: accountId } = useAuth();
  const mainRef = useRef(null);

  const [consented, setConsented] = useState(readConsent);
  const [userId, setUserId] = useState(accountId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [singleResult, setSingleResult] = useState(null); // { comparison, primary }
  const [batchRows, setBatchRows] = useState(null); // [{ id, text, status, result?, error? }]
  const [selectedBatchId, setSelectedBatchId] = useState(null);
  const [feedbackLog, setFeedbackLog] = useState([]);

  // Signing in/out switches identity: drop anything shown for the previous one.
  useEffect(() => {
    setUserId(accountId);
    setSingleResult(null);
    setBatchRows(null);
    setSelectedBatchId(null);
    setFeedbackLog([]);
    setError(null);
  }, [accountId]);

  // Page title + move focus to the new page for keyboard / screen-reader users.
  useEffect(() => {
    document.title = PAGE_TITLES[route] || "Page not found · MoodMentor";
    mainRef.current?.focus({ preventScroll: true });
  }, [route]);

  function updateConsent(value) {
    setConsented(value);
    try {
      if (value) localStorage.setItem(CONSENT_KEY, "yes");
      else localStorage.removeItem(CONSENT_KEY);
    } catch {
      // storage unavailable — consent just won't persist across reloads
    }
  }

  function addFeedback(entry) {
    setFeedbackLog((log) => [...log, entry]);
  }

  async function handleSingleSubmit({ text, userId: uid, modelType, topK }) {
    if (!consented) {
      setError("Please accept the privacy notice first.");
      return;
    }
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
    if (!consented) {
      setError("Please accept the privacy notice first.");
      return;
    }
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

  function renderAnalyze() {
    return (
      <>
        <ApiStatusBanner />
        <div className="layout">
          <AnalyzeForm
            onSubmitSingle={handleSingleSubmit}
            onSubmitBatch={handleBatchSubmit}
            isLoading={loading}
            defaultUserId={userId}
            lockUserId
            maxChars={MAX_CHARS}
            blockedReason={consented ? null : "Accept the privacy notice to enable analysis."}
          />

          <div className="feed">
            {!consented ? <ConsentCard checked={consented} onChange={updateConsent} /> : null}

            {error ? <div className="error-banner" role="alert">{error}</div> : null}

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
                {primaryResult ? <Insights result={primaryResult} /> : null}
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
                {primaryResult ? (
                  <GuidedReset
                    beforeResult={primaryResult}
                    recommendation={primaryResult.recommendations?.[0]}
                    userId={userId}
                    modelType={primaryResult.emotion_model}
                    onFeedback={addFeedback}
                  />
                ) : null}
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
                    <Insights result={selectedRow.result} />
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
      </>
    );
  }

  function renderRoute() {
    switch (route) {
      case "/":
        return <Landing />;
      case "/app":
        return renderAnalyze();
      case "/dashboard":
        return (
          <>
            <ApiStatusBanner />
            <div className="feed dashboard-wrap">
              <StatsPanel />
              <YearInPixels userId={userId} />
              <HistoryTrends userId={userId} />
            </div>
          </>
        );
      case "/login":
        return <AuthPage mode="login" navigate={navigate} />;
      case "/signup":
        return <AuthPage mode="signup" navigate={navigate} />;
      case "/about":
        return <About />;
      case "/privacy":
        return <Privacy />;
      case "/terms":
        return <Terms />;
      default:
        return <NotFound />;
    }
  }

  return (
    <div className="shell">
      <a
        className="skip-link"
        href="#/"
        onClick={(e) => {
          e.preventDefault();
          mainRef.current?.focus();
        }}
      >
        Skip to main content
      </a>
      <SiteHeader route={route} theme={theme} onToggleTheme={toggleTheme} />
      <main id="main" ref={mainRef} tabIndex={-1} className="main">
        <ErrorBoundary resetKey={route}>{renderRoute()}</ErrorBoundary>
      </main>
      <SiteFooter />
    </div>
  );
}
