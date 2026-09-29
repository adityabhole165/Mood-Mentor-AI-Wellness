"""Milestone 3 Part 2 - Task 8: dynamic recommendation explanations."""
from __future__ import annotations

from typing import Dict, List


def explain_recommendation(recommendation: dict, emotional_state: dict, trend: dict | None = None) -> dict:
    """Generate human-readable reasons from the actual ranking signals."""
    components = recommendation.get("components", {})
    reasons: List[str] = []

    emotion = emotional_state.get("dominant_emotion")
    intensity = float(emotional_state.get("intensity", 0))
    pref = float(components.get("user_preference", components.get("preference_score", 0)))
    semantic = float(components.get("content_similarity", 0))
    history = float(components.get("previous_interaction", components.get("history_affinity", 0)))
    emotion_rel = float(components.get("emotion_relevance", 0))

    if emotion and emotion_rel >= 0.35:
        reasons.append(
            f"High {emotion} relevance detected ({emotion_rel:.2f})."
        )
    if intensity >= 0.60:
        reasons.append(
            f"Emotion intensity is relatively high ({intensity:.2f})."
        )
    if pref >= 0.35:
        reasons.append(
            f"Matches the user's saved content preferences ({pref:.2f})."
        )
    if history >= 0.35:
        reasons.append(
            f"Related content has received positive historical interaction ({history:.2f})."
        )
    if semantic >= 0.35:
        reasons.append(
            f"Semantically matches the current wellness need ({semantic:.2f})."
        )

    if trend:
        repeated = trend.get("repeated_emotions", [])
        if emotion in repeated:
            reasons.append(
                f"{emotion} has appeared repeatedly in recent emotional history."
            )
        if trend.get("polarity_trend") == "worsening" and intensity >= 0.50:
            reasons.append("Recent emotional polarity has been trending downward.")

    if not reasons:
        reasons.append("Selected because the combined hybrid relevance score was positive.")

    return {
        "content_id": recommendation.get("content_id"),
        "title": recommendation.get("title"),
        "score": recommendation.get("score"),
        "reasons": reasons,
        "evidence": {
            "detected_emotion": emotion,
            "emotion_intensity": intensity,
            "user_preference": round(pref, 4),
            "historical_behavior": round(history, 4),
            "semantic_similarity": round(semantic, 4),
            "emotion_relevance": round(emotion_rel, 4),
            "trend": trend or {},
        },
    }


def add_explanations(recommendations: List[dict], emotional_state: dict, trend: dict) -> List[dict]:
    result = []
    for rec in recommendations:
        copy = dict(rec)
        explanation = explain_recommendation(copy, emotional_state, trend)
        copy["explanation"] = explanation
        copy["reasons"] = explanation["reasons"]
        result.append(copy)
    return result
