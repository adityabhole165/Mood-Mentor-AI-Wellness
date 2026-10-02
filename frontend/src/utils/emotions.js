// Shared emotion palette. Colour is never the only signal anywhere in the UI:
// every use also shows the emotion name (and usually an emoji).
export const EMOTION_META = {
  joy: { color: "#e0a21b", emoji: "😊", label: "Joy" },
  sadness: { color: "#4b79b8", emoji: "😢", label: "Sadness" },
  anger: { color: "#c4472f", emoji: "😠", label: "Anger" },
  fear: { color: "#7a5aa6", emoji: "😨", label: "Fear" },
  surprise: { color: "#2a9d8f", emoji: "😲", label: "Surprise" },
  disgust: { color: "#6b8e23", emoji: "🤢", label: "Disgust" },
};

export const EMOTIONS = Object.keys(EMOTION_META);

export function emotionColor(emotion) {
  return EMOTION_META[emotion]?.color || "#8a9086";
}

export function emotionEmoji(emotion) {
  return EMOTION_META[emotion]?.emoji || "•";
}

export function emotionLabel(emotion) {
  return EMOTION_META[emotion]?.label || emotion || "—";
}

export function clamp01(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

export function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Emotions the app treats as "negative" when judging whether a guided reset helped.
export const NEGATIVE_EMOTIONS = ["sadness", "anger", "fear", "disgust"];
