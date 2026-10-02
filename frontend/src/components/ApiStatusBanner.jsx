import { useEffect, useState } from "react";
import { getHealth } from "../api/client.js";

// Polls /health until the backend reports its models are loaded, so a cold
// start ("Space was asleep") reads as "waking up" instead of "broken".
export default function ApiStatusBanner() {
  const [status, setStatus] = useState({ kind: "checking" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer;

    async function poll() {
      try {
        const health = await getHealth();
        if (cancelled) return;
        const warm = String(health.model_warmup || "");
        if (warm === "ready") {
          setStatus({ kind: "ready" });
          return;
        }
        if (warm.startsWith("failed")) {
          setStatus({ kind: "model-failed", detail: warm });
          return;
        }
        setStatus({ kind: "warming" });
      } catch {
        if (cancelled) return;
        setStatus({ kind: "offline" });
      }
      timer = setTimeout(poll, 3000);
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [attempt]);

  if (status.kind === "ready") return null;

  if (status.kind === "checking") {
    return <div className="status-banner" role="status"><span className="spinner" /> Connecting to MoodMentor…</div>;
  }
  if (status.kind === "warming") {
    return (
      <div className="status-banner" role="status">
        <span className="spinner" /> Waking up the emotion models — your first analysis may take a little longer.
      </div>
    );
  }
  if (status.kind === "model-failed") {
    return (
      <div className="status-banner status-warn" role="alert">
        The server is up but the emotion models didn't load. Check that the model files or repo ids are configured.
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setStatus({ kind: "checking" }); setAttempt((a) => a + 1); }}>
          Retry
        </button>
      </div>
    );
  }
  return (
    <div className="status-banner status-warn" role="alert">
      Can't reach the MoodMentor server right now. It may be starting up.
      <button type="button" className="btn btn-ghost btn-sm" onClick={() => { setStatus({ kind: "checking" }); setAttempt((a) => a + 1); }}>
        Try again
      </button>
    </div>
  );
}
