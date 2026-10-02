import { EMOTIONS, clamp01, emotionColor, emotionLabel } from "./emotions.js";

// Builds a static, shareable PNG of the aura. It deliberately contains NO
// journal text — only the emotion mix, so sharing it doesn't leak anything private.
function buildSvg({ probabilities, dominant }) {
  const size = 1080;
  const cx = size / 2;
  const cy = 470;
  const defs = EMOTIONS.map(
    (e) =>
      `<radialGradient id="g-${e}"><stop offset="0%" stop-color="${emotionColor(e)}" stop-opacity="0.95"/><stop offset="100%" stop-color="${emotionColor(e)}" stop-opacity="0"/></radialGradient>`
  ).join("");
  const blobs = EMOTIONS.map((e, i) => {
    const p = clamp01(probabilities?.[e]);
    const angle = (i / EMOTIONS.length) * Math.PI * 2;
    const dist = 40 + (1 - p) * 110;
    const x = cx + Math.cos(angle) * dist;
    const y = cy + Math.sin(angle) * dist;
    const r = 150 + p * 330;
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="url(#g-${e})" opacity="${(0.25 + 0.75 * p).toFixed(2)}"/>`;
  }).join("");
  const date = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
<defs>${defs}</defs>
<rect width="${size}" height="${size}" fill="#1e2620"/>
${blobs}
<text x="${cx}" y="900" text-anchor="middle" font-family="Georgia, serif" font-size="84" fill="#fbfaf5">${emotionLabel(dominant)}</text>
<text x="${cx}" y="965" text-anchor="middle" font-family="Arial, sans-serif" font-size="34" fill="#aab4a8">${date}</text>
<text x="${cx}" y="1030" text-anchor="middle" font-family="Georgia, serif" font-size="36" fill="#8fc2b2">MoodMentor</text>
</svg>`;
}

export async function downloadAuraCard({ probabilities, dominant }) {
  const svg = buildSvg({ probabilities, dominant });
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
  try {
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("Could not render the card."));
      img.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = 1080;
    canvas.height = 1080;
    canvas.getContext("2d").drawImage(img, 0, 0);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new Error("Could not create the image.");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `moodmentor-aura-${new Date().toISOString().slice(0, 10)}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  } finally {
    URL.revokeObjectURL(url);
  }
}
