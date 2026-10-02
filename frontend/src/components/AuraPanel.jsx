import { useState } from "react";
import EmotionAura from "./EmotionAura.jsx";
import { downloadAuraCard } from "../utils/auraCard.js";
import { emotionLabel } from "../utils/emotions.js";

export default function AuraPanel({ result }) {
  const state = result?.emotional_state || {};
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function save() {
    setBusy(true);
    setError(null);
    try {
      await downloadAuraCard({ probabilities: state.emotion_probabilities, dominant: state.dominant_emotion });
    } catch (err) {
      setError(err.message || "Could not save the card.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel aura-panel">
      <EmotionAura
        probabilities={state.emotion_probabilities}
        intensity={state.intensity}
        dominant={state.dominant_emotion}
        size={220}
        showLabel={false}
      />
      <div className="aura-panel-copy">
        <h2 style={{ fontSize: 18 }}>Your emotional aura</h2>
        <p className="hint" style={{ margin: "4px 0 10px" }}>
          Each colour is an emotion; bigger and brighter means stronger. The faster it breathes, the more intense the
          entry reads. Right now it's mostly <strong>{emotionLabel(state.dominant_emotion)}</strong>.
        </p>
        <button type="button" className="btn btn-ghost btn-sm" onClick={save} disabled={busy}>
          {busy ? <span className="spinner" /> : "⬇"} Save aura card
        </button>
        <p className="hint" style={{ margin: "6px 0 0" }}>The card contains only the colours and date — never your text.</p>
        {error ? <div className="error-banner" role="alert" style={{ marginTop: 8 }}>{error}</div> : null}
      </div>
    </div>
  );
}
