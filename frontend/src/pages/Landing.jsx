import { useEffect, useState } from "react";
import EmotionAura from "../components/EmotionAura.jsx";

// Sample emotion mixes the hero aura cycles through (purely illustrative).
const SAMPLES = [
  { dominant: "joy", intensity: 0.45, probabilities: { joy: 0.82, surprise: 0.3, fear: 0.08, sadness: 0.05, anger: 0.03, disgust: 0.02 } },
  { dominant: "fear", intensity: 0.8, probabilities: { fear: 0.74, joy: 0.45, surprise: 0.2, sadness: 0.18, anger: 0.06, disgust: 0.04 } },
  { dominant: "sadness", intensity: 0.3, probabilities: { sadness: 0.78, fear: 0.25, joy: 0.06, anger: 0.1, surprise: 0.04, disgust: 0.07 } },
  { dominant: "anger", intensity: 0.9, probabilities: { anger: 0.85, disgust: 0.35, sadness: 0.2, fear: 0.1, joy: 0.02, surprise: 0.12 } },
];

const STEPS = [
  { n: "1", title: "Write or upload", body: "Type a journal entry, or bring a .txt or .csv file. Nothing is sent until you press analyze." },
  { n: "2", title: "Two models read it", body: "VADER, BERT and DistilBERT score six emotions — joy, sadness, anger, fear, surprise and disgust — with confidence." },
  { n: "3", title: "Get a gentle next step", body: "A hybrid recommender ranks wellness activities for you and explains exactly why each was chosen." },
];

const FEATURES = [
  { icon: "🔍", title: "See why", body: "Highlight the words that pushed the model toward an emotion, instead of trusting a black box." },
  { icon: "🌬️", title: "Guided resets that learn", body: "Do a 60-second exercise, re-check in, and MoodMentor measures whether it helped — then adapts." },
  { icon: "🟩", title: "Year in pixels", body: "Every day becomes a coloured square, so patterns in your mood become visible at a glance." },
  { icon: "🧭", title: "Personalised & explained", body: "Rankings blend your emotions, intensity, preferences and feedback — and every card shows its reasoning." },
  { icon: "📈", title: "Trends & reports", body: "Daily, weekly and monthly trends, plus CSV and JSON exports of your own data." },
  { icon: "🔒", title: "Yours to delete", body: "One click removes everything stored under your ID. No ads, no selling data." },
];

export default function Landing() {
  const [i, setI] = useState(0);

  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return undefined;
    const id = setInterval(() => setI((n) => (n + 1) % SAMPLES.length), 4200);
    return () => clearInterval(id);
  }, []);

  const sample = SAMPLES[i];

  return (
    <div className="landing">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Emotion-aware journaling</p>
          <h2 className="hero-title serif">Understand how you feel — and what might help.</h2>
          <p className="hero-sub">
            Write a few honest sentences. MoodMentor reads the emotional tone, shows you what drove it, and suggests
            small, personalised wellness steps that get smarter every time you use them.
          </p>
          <div className="hero-cta">
            <a className="btn btn-primary" href="#/app" style={{ width: "auto" }}>Try it now</a>
            <a className="btn btn-ghost" href="#how-it-works" onClick={(e) => { e.preventDefault(); document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" }); }}>
              How it works
            </a>
          </div>
          <p className="hint" style={{ marginTop: 12 }}>
            No account needed. Educational tool — not medical advice.
          </p>
        </div>
        <div className="hero-visual">
          <EmotionAura
            probabilities={sample.probabilities}
            intensity={sample.intensity}
            dominant={sample.dominant}
            size={300}
          />
          <p className="hint" style={{ textAlign: "center" }}>Example aura — yours is generated from your own entry.</p>
        </div>
      </section>

      <section id="how-it-works" className="section" aria-labelledby="how-title">
        <h2 id="how-title" className="serif section-h">How it works</h2>
        <ol className="steps">
          {STEPS.map((s) => (
            <li key={s.n} className="panel step">
              <span className="step-n serif" aria-hidden="true">{s.n}</span>
              <h3 style={{ fontSize: 17, margin: "6px 0" }}>{s.title}</h3>
              <p style={{ margin: 0, color: "var(--ink-soft)" }}>{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="section" aria-labelledby="features-title">
        <h2 id="features-title" className="serif section-h">What makes it different</h2>
        <div className="feature-grid">
          {FEATURES.map((f) => (
            <div key={f.title} className="panel feature">
              <div className="feature-icon" aria-hidden="true">{f.icon}</div>
              <h3 style={{ fontSize: 16, margin: "6px 0 4px" }}>{f.title}</h3>
              <p style={{ margin: 0, color: "var(--ink-soft)", fontSize: 14 }}>{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section cta-band">
        <h2 className="serif" style={{ fontSize: 24, margin: "0 0 8px" }}>Ready for a two-minute check-in?</h2>
        <p style={{ margin: "0 0 16px", color: "var(--ink-soft)" }}>Your first analysis takes a few seconds.</p>
        <a className="btn btn-primary" href="#/app" style={{ width: "auto" }}>Start a check-in</a>
      </section>
    </div>
  );
}
