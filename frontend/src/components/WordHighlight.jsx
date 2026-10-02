import { useState } from "react";
import { ApiError, explainText } from "../api/client.js";
import { clamp01, emotionColor, emotionLabel, hexToRgba } from "../utils/emotions.js";

// "Why this emotion?" — asks the backend to remove each word in turn and
// measure how much the target emotion's probability drops. Words whose
// removal hurts the most are highlighted (and underlined, so it isn't colour-only).
export default function WordHighlight({ text, modelType, target }) {
  const [state, setState] = useState({ status: "idle" });
  const color = emotionColor(target);

  async function run() {
    setState({ status: "loading" });
    try {
      const data = await explainText({ text, modelType, targetEmotion: target });
      setState({ status: "done", data });
    } catch (err) {
      if (err instanceof ApiError && (err.status === 404 || err.status === 501)) {
        setState({ status: "unavailable" });
      } else {
        setState({ status: "error", message: err.message || "Could not explain this entry." });
      }
    }
  }

  const words = state.status === "done" ? state.data.words || [] : [];
  const drivers = [...words]
    .filter((w) => w.weight >= 0.35)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 5)
    .map((w) => w.word.replace(/[^\p{L}\p{N}'-]/gu, ""))
    .filter(Boolean);

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h2 style={{ fontSize: 16 }}>Why this emotion?</h2>
          <p className="hint" style={{ margin: "2px 0 0" }}>
            See which words pushed the model toward <strong>{emotionLabel(target)}</strong>.
          </p>
        </div>
        <button type="button" className="btn btn-ghost btn-sm" onClick={run} disabled={state.status === "loading"}>
          {state.status === "loading" ? <span className="spinner" /> : null}
          {state.status === "done" ? "Re-run" : "Highlight words"}
        </button>
      </div>

      {state.status === "unavailable" ? (
        <p className="not-available" style={{ marginTop: 12 }}>
          Word-level explanations need the optional <code>/explain</code> endpoint on the backend (see backend_extras).
        </p>
      ) : null}
      {state.status === "error" ? <div className="error-banner" role="alert" style={{ marginTop: 12 }}>{state.message}</div> : null}

      {state.status === "done" ? (
        <>
          <p className="highlight-text" aria-label="Entry with influential words highlighted">
            {words.map((w, i) => {
              const weight = clamp01(w.weight);
              const strong = weight >= 0.5;
              return (
                <span key={i}>
                  <mark
                    className={strong ? "hl hl-strong" : "hl"}
                    style={{ background: weight > 0.05 ? hexToRgba(color, 0.12 + 0.55 * weight) : "transparent" }}
                    title={weight > 0.05 ? `Removing this word lowers ${emotionLabel(target)} by ${Math.round((w.delta ?? 0) * 100)} points` : undefined}
                  >
                    {w.word}
                  </mark>{" "}
                </span>
              );
            })}
          </p>
          <p className="hint" style={{ marginBottom: 0 }}>
            {drivers.length
              ? `Strongest drivers: ${drivers.join(", ")}.`
              : "No single word dominated — the emotion comes from the sentence as a whole."}{" "}
            Highlights show what this model reacted to, not what you meant.
          </p>
        </>
      ) : null}
    </div>
  );
}
