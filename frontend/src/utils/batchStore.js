// Remembers every statement analysed from .txt / .csv uploads (per user) in
// this browser, so the Dashboard can graph all evaluated data even after the
// Analyze page is left or reloaded, and regardless of backend history support.

const KEY_PREFIX = "mm_batch_analysed_";
const MAX_ENTRIES = 1000;

function key(userId) {
  return `${KEY_PREFIX}${userId || "demo_user"}`;
}

export function loadBatchEntries(userId) {
  try {
    const raw = localStorage.getItem(key(userId));
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function appendBatchEntry(userId, entry) {
  try {
    const list = loadBatchEntries(userId).filter((e) => e.id !== entry.id);
    list.push(entry);
    localStorage.setItem(key(userId), JSON.stringify(list.slice(-MAX_ENTRIES)));
    window.dispatchEvent(new Event("mm-batch-updated"));
  } catch {
    // storage unavailable — dashboard just won't include this entry
  }
}

export function clearBatchEntries(userId) {
  try {
    localStorage.removeItem(key(userId));
    window.dispatchEvent(new Event("mm-batch-updated"));
  } catch {
    // ignore
  }
}

// Compact summary of one finished analysis row.
export function toBatchEntry(runId, row, source) {
  const state = row.result?.emotional_state || {};
  return {
    id: `${runId}-${row.id}`,
    timestamp: new Date().toISOString(),
    source,
    text: String(row.text || "").slice(0, 200),
    dominant_emotion: state.dominant_emotion || row.result?.confidence?.primary_emotion || null,
    intensity: state.intensity ?? null,
    confidence: state.emotion_confidence ?? row.result?.confidence?.primary_confidence ?? null,
    polarity_score: state.polarity_score ?? row.result?.sentiment?.compound ?? null,
  };
}
