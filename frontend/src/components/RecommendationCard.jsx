import { useEffect, useRef, useState } from "react";
import { sendFeedback } from "../api/client.js";

const ACTIONS = [
  { event: "helpful", label: "👍 Helpful", cls: "btn-accept" },
  { event: "complete", label: "✅ Complete", cls: "btn-accept" },
  { event: "accept", label: "✓ Accept", cls: "btn-accept" },
  { event: "reject", label: "✕ Not for me", cls: "btn-reject" },
  { event: "skip", label: "⏭ Skip", cls: "btn-view" },
  { event: "hide", label: "🙈 Hide", cls: "btn-view" },
];

const COMPONENT_LABELS = {
  emotion_relevance: "emotion relevance",
  user_preference: "user preference",
  content_similarity: "semantic similarity",
  collaborative_score: "collaborative",
  previous_interaction: "history affinity",
  rule_score: "rule score",
  personalized_ml_score: "personalized ML",
  feedback_ml_score: "feedback ML",
};

export default function RecommendationCard({ rec, userId, onFeedback }) {
  const [lastEvent, setLastEvent] = useState(null);
  const [rating, setRating] = useState(0);
  const [tags, setTags] = useState("");
  const [note, setNote] = useState("");
  const [showBreakdown, setShowBreakdown] = useState(false);
  const viewSent = useRef(false);

  useEffect(() => {
    if (viewSent.current) return;
    viewSent.current = true;
    sendFeedback({ userId, contentId: rec.content_id, event: "view" }).catch(() => {});
  }, [rec.content_id, userId]);

  async function react(event) {
    try {
      await sendFeedback({ userId, contentId: rec.content_id, event });
      setLastEvent(event);
      setNote(`Logged "${event}"`);
      onFeedback?.({ contentId: rec.content_id, title: rec.title, event, timestamp: new Date().toISOString() });
    } catch (err) {
      setNote(err.message || "Could not send feedback");
    }
  }

  async function saveRating() {
    const preferenceChanges = tags.trim()
      ? { preferred_tags: tags.split(",").map((t) => t.trim()).filter(Boolean) }
      : {};
    try {
      await sendFeedback({
        userId,
        contentId: rec.content_id,
        event: "rate",
        rating,
        preferenceChanges,
      });
      setNote(`Rated ${rating}/5${tags.trim() ? " · preferences saved" : ""}`);
      onFeedback?.({ contentId: rec.content_id, title: rec.title, event: "rate", timestamp: new Date().toISOString() });
    } catch (err) {
      setNote(err.message || "Could not send rating");
    }
  }

  const reasons = rec.explanation?.reasons || rec.reasons || [];
  const components = rec.components || {};
  const componentEntries = Object.entries(COMPONENT_LABELS)
    .filter(([key]) => key in components)
    .map(([key, label]) => [label, Number(components[key]) || 0]);

  return (
    <div className="rec-card">
      <div className="rec-head">
        <span className="rec-rank">{String(rec.rank).padStart(2, "0")}</span>
        <span className="rec-title serif">{rec.title}</span>
        <span className="rec-score">score {Number(rec.score).toFixed(2)}</span>
      </div>

      {reasons.length ? (
        <ul className="rec-reasons">
          {reasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      ) : null}

      <div className="rec-actions">
        {ACTIONS.map((a) => (
          <button
            key={a.event}
            className={`btn btn-sm ${a.cls}`}
            onClick={() => react(a.event)}
            disabled={lastEvent === a.event}
          >
            {lastEvent === a.event ? "Logged" : a.label}
          </button>
        ))}
        {rec.url ? (
          <a className="btn btn-sm btn-view" href={rec.url} target="_blank" rel="noreferrer">
            Open
          </a>
        ) : null}
        <button
          type="button"
          className="btn btn-sm btn-view"
          style={{ marginLeft: "auto" }}
          onClick={() => setShowBreakdown((v) => !v)}
        >
          {showBreakdown ? "Hide breakdown" : "Score breakdown"}
        </button>
      </div>

      <div className="rec-actions" style={{ marginTop: 6 }}>
        <span className="rating-stars">
          {[1, 2, 3, 4, 5].map((n) => (
            <button
              key={n}
              type="button"
              className={n <= rating ? "filled" : ""}
              onClick={() => setRating(n)}
              aria-label={`Rate ${n} out of 5`}
            >
              ★
            </button>
          ))}
        </span>
        <input
          placeholder="preferred tags, comma separated"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          style={{
            flex: 1,
            minWidth: 160,
            padding: "5px 9px",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            fontSize: 12.5,
          }}
        />
        <button type="button" className="btn btn-sm btn-ghost" onClick={saveRating} disabled={!rating}>
          Save rating
        </button>
      </div>

      {note ? <p className="feedback-note">{note}</p> : null}

      {showBreakdown && componentEntries.length ? (
        <div className="bars" style={{ marginTop: 12 }}>
          {componentEntries.map(([label, value]) => (
            <div className="bar-row" key={label}>
              <span className="label">{label}</span>
              <span className="bar-track">
                <span className="bar-fill" style={{ width: `${Math.min(100, Math.round(value * 100))}%` }} />
              </span>
              <span className="value">{value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
