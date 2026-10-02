import { useEffect, useRef, useState } from "react";
import { analyzeText, sendFeedback } from "../api/client.js";
import { NEGATIVE_EMOTIONS, emotionLabel } from "../utils/emotions.js";

const SESSION_SECONDS = 60;

// pattern: [label, seconds, kind]  kind: in | hold | out
const RESETS = {
  fear: {
    name: "Box breathing",
    pattern: [["Breathe in", 4, "in"], ["Hold", 4, "hold"], ["Breathe out", 4, "out"], ["Hold", 4, "hold"]],
    prompt: "While you breathe, name five things you can see around you.",
  },
  anger: {
    name: "Long-exhale breathing",
    pattern: [["Breathe in", 4, "in"], ["Breathe out", 6, "out"]],
    prompt: "Unclench your jaw and let your shoulders drop on every exhale.",
  },
  sadness: {
    name: "Gentle grounding",
    pattern: [["Breathe in", 4, "in"], ["Hold", 2, "hold"], ["Breathe out", 6, "out"]],
    prompt: "Think of one small thing from today that felt even a little okay.",
  },
  disgust: {
    name: "Reset breathing",
    pattern: [["Breathe in", 4, "in"], ["Breathe out", 6, "out"]],
    prompt: "Let your eyes rest on something neutral or pleasant while you breathe.",
  },
  surprise: {
    name: "Settling breaths",
    pattern: [["Breathe in", 4, "in"], ["Hold", 2, "hold"], ["Breathe out", 4, "out"]],
    prompt: "Feel your feet on the floor and let the moment settle.",
  },
  joy: {
    name: "Savoring breaths",
    pattern: [["Breathe in", 4, "in"], ["Hold", 4, "hold"], ["Breathe out", 4, "out"]],
    prompt: "What made this feel good? Hold on to it for a few breaths.",
  },
};

const MIN_SCALE = 0.55;

function pacerAt(pattern, elapsed) {
  const cycle = pattern.reduce((sum, p) => sum + p[1], 0);
  let t = elapsed % cycle;
  let level = MIN_SCALE; // scale at the start of the cycle
  for (const [label, secs, kind] of pattern) {
    if (t < secs) {
      const f = t / secs;
      if (kind === "in") return { label, scale: MIN_SCALE + (1 - MIN_SCALE) * f, remaining: Math.ceil(secs - t) };
      if (kind === "out") return { label, scale: 1 - (1 - MIN_SCALE) * f, remaining: Math.ceil(secs - t) };
      return { label, scale: level, remaining: Math.ceil(secs - t) };
    }
    t -= secs;
    level = kind === "in" ? 1 : kind === "out" ? MIN_SCALE : level;
  }
  return { label: pattern[0][0], scale: MIN_SCALE, remaining: 0 };
}

// How much better does the user feel? Positive = improvement.
function improvement(emotion, beforeState, afterState) {
  const before = beforeState?.emotion_probabilities?.[emotion] ?? 0;
  const after = afterState?.emotion_probabilities?.[emotion] ?? 0;
  if (NEGATIVE_EMOTIONS.includes(emotion)) return { shift: before - after, before, after };
  const pb = beforeState?.polarity_score ?? 0;
  const pa = afterState?.polarity_score ?? 0;
  return { shift: pa - pb, before, after };
}

function ratingFromShift(shift) {
  if (shift >= 0.3) return 5;
  if (shift >= 0.15) return 4;
  if (shift >= 0.03) return 3;
  if (shift >= -0.05) return 2;
  return 1;
}

export default function GuidedReset({ beforeResult, recommendation, userId, modelType, onFeedback }) {
  const state = beforeResult?.emotional_state || {};
  const emotion = state.dominant_emotion && RESETS[state.dominant_emotion] ? state.dominant_emotion : "fear";
  const reset = RESETS[emotion];

  const [stage, setStage] = useState("intro"); // intro | running | checkout | done
  const [elapsed, setElapsed] = useState(0);
  const [afterText, setAfterText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [outcome, setOutcome] = useState(null);
  const startedAt = useRef(0);

  // A new analysis means a new reset.
  useEffect(() => {
    setStage("intro");
    setElapsed(0);
    setAfterText("");
    setError(null);
    setOutcome(null);
  }, [beforeResult]);

  useEffect(() => {
    if (stage !== "running") return undefined;
    startedAt.current = Date.now() - elapsed * 1000;
    const id = setInterval(() => {
      const secs = (Date.now() - startedAt.current) / 1000;
      if (secs >= SESSION_SECONDS) {
        setElapsed(SESSION_SECONDS);
        setStage("checkout");
      } else {
        setElapsed(secs);
      }
    }, 100);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage]);

  const pacer = pacerAt(reset.pattern, elapsed);

  async function submitCheckout(e) {
    e.preventDefault();
    if (!afterText.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const after = await analyzeText({
        text: afterText.trim(),
        userId,
        modelType,
        topK: 1,
        persistHistory: false,
      });
      const { shift, before, after: afterProb } = improvement(emotion, state, after.emotional_state);
      const rating = ratingFromShift(shift);

      let feedbackSaved = false;
      if (recommendation?.content_id) {
        try {
          await sendFeedback({ userId, contentId: recommendation.content_id, event: "complete" });
          await sendFeedback({ userId, contentId: recommendation.content_id, event: "rate", rating });
          feedbackSaved = true;
          onFeedback?.({
            contentId: recommendation.content_id,
            title: recommendation.title,
            event: `reset outcome (${rating}/5)`,
            timestamp: new Date().toISOString(),
          });
        } catch {
          feedbackSaved = false;
        }
      }
      setOutcome({ shift, before, after: afterProb, rating, feedbackSaved, afterEmotion: after.emotional_state?.dominant_emotion });
      setStage("done");
    } catch (err) {
      setError(err.message || "Could not read your check-out.");
    } finally {
      setBusy(false);
    }
  }

  function restart() {
    setStage("intro");
    setElapsed(0);
    setAfterText("");
    setOutcome(null);
    setError(null);
  }

  const pct = (n) => `${Math.round(n * 100)}%`;
  const remainingSeconds = Math.max(0, Math.ceil(SESSION_SECONDS - elapsed));

  return (
    <div className="panel guided-reset">
      <div className="panel-head">
        <div>
          <h2 style={{ fontSize: 16 }}>60-second guided reset</h2>
          <p className="hint" style={{ margin: "2px 0 0" }}>
            {reset.name} for {emotionLabel(emotion).toLowerCase()}. Afterwards MoodMentor re-reads how you feel and
            learns whether it actually helped.
          </p>
        </div>
      </div>

      {stage === "intro" ? (
        <div className="reset-intro">
          <p style={{ margin: "0 0 12px" }}>{reset.prompt}</p>
          <button type="button" className="btn btn-primary" style={{ width: "auto" }} onClick={() => setStage("running")}>
            Start reset
          </button>
        </div>
      ) : null}

      {stage === "running" ? (
        <div className="reset-run">
          <div className="pacer-stage" aria-hidden="true">
            <div className="pacer" style={{ transform: `scale(${pacer.scale})` }} />
          </div>
          <div className="pacer-text" role="timer" aria-live="off">
            <strong className="serif">{pacer.label}</strong>
            <span className="hint"> · {pacer.remaining}s</span>
          </div>
          <p className="hint" style={{ textAlign: "center", margin: "6px 0 12px" }}>{reset.prompt}</p>
          <div className="reset-foot">
            <span className="hint">{remainingSeconds}s left</span>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setStage("checkout")}>
              Finish early
            </button>
          </div>
        </div>
      ) : null}

      {stage === "checkout" ? (
        <form onSubmit={submitCheckout} className="reset-checkout">
          <div className="field">
            <label htmlFor="afterText">In one line — how do you feel now?</label>
            <textarea
              id="afterText"
              value={afterText}
              maxLength={500}
              onChange={(e) => setAfterText(e.target.value)}
              placeholder="A little calmer, but still thinking about…"
              style={{ minHeight: 70 }}
            />
          </div>
          {error ? <div className="error-banner" role="alert">{error}</div> : null}
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-primary" style={{ width: "auto" }} type="submit" disabled={busy || !afterText.trim()}>
              {busy ? <span className="spinner" /> : null}
              {busy ? "Reading…" : "See if it helped"}
            </button>
            <button type="button" className="btn btn-ghost" onClick={restart}>Skip</button>
          </div>
        </form>
      ) : null}

      {stage === "done" && outcome ? (
        <div className="reset-done" aria-live="polite">
          <p className="serif" style={{ fontSize: 20, margin: "0 0 6px" }}>
            {outcome.shift >= 0.03 ? "That seems to have helped." : outcome.shift >= -0.05 ? "About the same — that's okay." : "Not this time — that's useful to know."}
          </p>
          <p style={{ margin: "0 0 8px" }}>
            {NEGATIVE_EMOTIONS.includes(emotion)
              ? `${emotionLabel(emotion)} went from ${pct(outcome.before)} to ${pct(outcome.after)}.`
              : `Overall tone shifted by ${outcome.shift >= 0 ? "+" : ""}${outcome.shift.toFixed(2)}.`}
          </p>
          <p className="hint" style={{ margin: "0 0 12px" }}>
            {outcome.feedbackSaved
              ? `Saved as a ${outcome.rating}/5 rating for “${recommendation.title}”, so future suggestions can learn from it.`
              : "Not saved as feedback (no recommendation to attach it to, or the server was unreachable)."}
          </p>
          <button type="button" className="btn btn-ghost btn-sm" onClick={restart}>Do another reset</button>
        </div>
      ) : null}
    </div>
  );
}
