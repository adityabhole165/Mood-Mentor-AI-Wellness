import { useId } from "react";
import { EMOTIONS, clamp01, emotionColor, emotionEmoji, emotionLabel } from "../utils/emotions.js";

// Generative "aura": one soft blob per emotion, sized and opacified by its
// probability, all breathing at a speed driven by emotional intensity.
export default function EmotionAura({ probabilities = {}, intensity = 0.4, dominant, size = 260, showLabel = true }) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const dur = (7 - 4 * clamp01(intensity)).toFixed(2); // 3s (intense) … 7s (calm)

  const blobs = EMOTIONS.map((emotion, i) => {
    const p = clamp01(probabilities[emotion]);
    const angle = (i / EMOTIONS.length) * Math.PI * 2;
    const dist = 12 + (1 - p) * 38;
    return {
      emotion,
      p,
      cx: 150 + Math.cos(angle) * dist,
      cy: 150 + Math.sin(angle) * dist,
      r: 45 + p * 95,
      delay: -(i * 0.9),
    };
  });

  const summary = blobs
    .filter((b) => b.p >= 0.05)
    .sort((a, b) => b.p - a.p)
    .map((b) => `${emotionLabel(b.emotion)} ${Math.round(b.p * 100)}%`)
    .join(", ");

  return (
    <figure className="aura" style={{ "--aura-size": `${size}px`, "--aura-dur": `${dur}s` }}>
      <svg
        viewBox="0 0 300 300"
        role="img"
        aria-label={`Emotion aura. ${summary || "No strong emotion detected"}. Intensity ${Math.round(clamp01(intensity) * 100)} percent.`}
      >
        <defs>
          {EMOTIONS.map((e) => (
            <radialGradient id={`${uid}-${e}`} key={e}>
              <stop offset="0%" stopColor={emotionColor(e)} stopOpacity="0.95" />
              <stop offset="100%" stopColor={emotionColor(e)} stopOpacity="0" />
            </radialGradient>
          ))}
        </defs>
        {blobs.map((b) => (
          <circle
            key={b.emotion}
            className="aura-blob"
            style={{ animationDelay: `${b.delay}s` }}
            cx={b.cx}
            cy={b.cy}
            r={b.r}
            fill={`url(#${uid}-${b.emotion})`}
            opacity={0.25 + 0.75 * b.p}
          />
        ))}
      </svg>
      {showLabel ? (
        <figcaption className="aura-caption">
          <span className="aura-emoji" aria-hidden="true">{emotionEmoji(dominant)}</span>{" "}
          <span className="serif">{emotionLabel(dominant)}</span>
        </figcaption>
      ) : null}
    </figure>
  );
}
