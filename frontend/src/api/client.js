// Thin fetch wrapper around the MoodMentor FastAPI backend.
//
// Core endpoints (present on every backend build):
//   GET  /health
//   POST /analyze
//   POST /feedback
//   GET  /stats
//
// Extended endpoints (only present on the "full" M4 backend build). Every
// call to one of these is wrapped so a 404 on a leaner backend degrades
// gracefully instead of crashing the UI:
//   GET    /history/emotions
//   GET    /history/recommendations
//   GET    /history/feedback
//   GET    /trends
//   GET    /reports
//   DELETE /users/{user_id}/data

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new ApiError(
      `Could not reach the MoodMentor API at ${BASE_URL}. Is the backend running?`,
      0
    );
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      // ignore body parse failure, fall back to statusText
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return null;
  return response.json();
}

// --- Core endpoints -------------------------------------------------------

export function getHealth() {
  return request("/health");
}

export function analyzeText({
  text,
  userId = "demo_user",
  topK = 5,
  modelType = "BERT",
  preferredTypes = [],
  preferredTags = [],
  blockedContentIds = [],
  persistHistory = true,
}) {
  return request("/analyze", {
    method: "POST",
    body: JSON.stringify({
      text,
      user_id: userId,
      top_k: topK,
      model_type: modelType,
      preferred_types: preferredTypes,
      preferred_tags: preferredTags,
      blocked_content_ids: blockedContentIds,
      persist_history: persistHistory,
    }),
  });
}

export function sendFeedback({ userId, contentId, event, rating = null, preferenceChanges = {} }) {
  return request("/feedback", {
    method: "POST",
    body: JSON.stringify({
      user_id: userId,
      content_id: contentId,
      event,
      rating,
      preference_changes: preferenceChanges,
    }),
  });
}

export function getStats() {
  return request("/stats");
}

// Runs the same text through both emotion models in parallel, mirroring the
// Streamlit app's "BERT vs DistilBERT" side-by-side view. `persistHistory`
// defaults to false for the second call so a single check-in doesn't write
// two emotion-history rows.
export async function compareModels(baseArgs) {
  const [bert, distilbert] = await Promise.all([
    analyzeText({ ...baseArgs, modelType: "BERT" }),
    analyzeText({ ...baseArgs, modelType: "DistilBERT", persistHistory: false }),
  ]);
  return {
    bert,
    distilbert,
    agreeOnPrimary: bert.confidence?.primary_emotion === distilbert.confidence?.primary_emotion,
  };
}

// --- Extended endpoints (optional backend) --------------------------------
// These resolve to { available: false } instead of throwing when the
// endpoint isn't implemented on the connected backend (404/501), so the UI
// can show an inline "not available on this backend" note rather than break.

async function optionalRequest(path) {
  try {
    const data = await request(path);
    return { available: true, data };
  } catch (err) {
    if (err instanceof ApiError && (err.status === 404 || err.status === 501)) {
      return { available: false, data: null };
    }
    throw err;
  }
}

export function getEmotionHistory(userId) {
  return optionalRequest(`/history/emotions?user_id=${encodeURIComponent(userId)}`);
}

export function getRecommendationHistory(userId) {
  return optionalRequest(`/history/recommendations?user_id=${encodeURIComponent(userId)}`);
}

export function getFeedbackHistory(userId) {
  return optionalRequest(`/history/feedback?user_id=${encodeURIComponent(userId)}`);
}

export function getTrends(userId, granularity = "daily") {
  return optionalRequest(
    `/trends?user_id=${encodeURIComponent(userId)}&granularity=${granularity}`
  );
}

export function getReportUrl(userId, format = "csv") {
  return `${BASE_URL}/reports?user_id=${encodeURIComponent(userId)}&format=${format}`;
}

export function deleteUserData(userId) {
  return request(`/users/${encodeURIComponent(userId)}/data`, { method: "DELETE" });
}

export { ApiError, BASE_URL };
