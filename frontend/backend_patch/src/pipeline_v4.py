"""Milestone 3 Part 2 canonical end-to-end pipeline (Tasks 6-10)."""
from __future__ import annotations
import os
from typing import Optional

from .ingestion import read_input_data
from .preprocessing import preprocess_text
from .sentiment import analyze_sentiment
from . import emotion_bert, emotion_distilbert
from .model_cache import get_emotion_model, get_semantic_matcher
from .confidence import build_confidence_report
from .emotional_state import analyze_emotional_state
from .recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile
from .data_loader import load_wellness_content, load_interactions
from .hybrid_recommender import HybridRecommendationEngine
from .ranking import RecommendationRanker
from .personalized_recommender import train_personalized_model, FEATURE_NAMES, feature_dict
from .emotion_history import load_emotion_history, append_emotion_history, analyze_emotion_trend, build_history_record, history_influence
from .recommendation_feedback import load_feedback, train_model_from_feedback, feedback_summary
from .recommendation_explainability import add_explanations


def run_milestone3_part2(
    text: str,
    user_profile: UserProfile,
    *,
    interactions=None,
    contents=None,
    emotion_model_dir: Optional[str] = None,
    model_type: str = "BERT",
    wellness_content_csv: Optional[str] = None,
    interactions_csv: Optional[str] = None,
    emotion_history_csv: Optional[str] = None,
    feedback_csv: Optional[str] = None,
    top_k: int = 5,
    semantic_model_name: str = "all-MiniLM-L6-v2",
    persist_history: bool = True,
) -> dict:
    """Run the full M3 Part 2 path without UI-specific state.

    Feedback is deliberately a separate model. It is blended by the hybrid
    engine with the interaction model (65% / 35%) once enough valid labels
    exist, so Task 7 cannot silently replace the base personalization model.
    """
    if wellness_content_csv and os.path.exists(wellness_content_csv):
        contents = load_wellness_content(wellness_content_csv)
    contents = contents or DEFAULT_WELLNESS_CONTENT
    if interactions is None and interactions_csv:
        interactions = load_interactions(interactions_csv)
    interactions = interactions or []
    model_dir = emotion_model_dir or emotion_bert.DEFAULT_SAVE_DIR

    record = read_input_data("raw_text", text)[0]
    if not record.is_valid:
        return {"is_valid": False, "error": record.error, "recommendations": []}

    processed = preprocess_text(record.raw_text)
    sentiment = analyze_sentiment(processed.cleaned_text)

    model_type = model_type.upper()
    if model_type == "DISTILBERT":
        model, tokenizer = get_emotion_model(
            "DISTILBERT", emotion_model_dir or emotion_distilbert.DEFAULT_SAVE_DIR
        )
        prediction = emotion_distilbert.predict(processed.cleaned_text, model, tokenizer)
    elif model_type == "BERT":
        model, tokenizer = get_emotion_model("BERT", model_dir)
        prediction = emotion_bert.predict(processed.cleaned_text, model, tokenizer)
    else:
        raise ValueError("model_type must be 'BERT' or 'DistilBERT'")

    confidence = build_confidence_report(prediction.scores)
    state = analyze_emotional_state(confidence.all_scores, sentiment.compound)

    history = load_emotion_history(emotion_history_csv, user_profile.user_id) if emotion_history_csv else []
    trend = analyze_emotion_trend(history)
    hist_affinity = history_influence(history, state.dominant_emotion)

    matcher = get_semantic_matcher(semantic_model_name, contents)
    semantic_scores = matcher.match_emotional_state(state, record.raw_text, contents)

    base_model = train_personalized_model(interactions)
    feedback_records = load_feedback(feedback_csv, user_profile.user_id) if feedback_csv else []
    feedback_model = train_model_from_feedback(feedback_records) if feedback_records else None

    # Persist preference changes from feedback into the current profile.
    preferred_tags = set(user_profile.preferred_tags)
    preferred_types = set(user_profile.preferred_types)
    for fb in feedback_records:
        changes = fb.preference_changes or {}
        preferred_tags.update(str(x) for x in changes.get("preferred_tags", []) if x)
        preferred_types.update(str(x) for x in changes.get("preferred_types", []) if x)
    effective_profile = UserProfile(
        user_id=user_profile.user_id,
        preferred_types=sorted(preferred_types),
        preferred_tags=sorted(preferred_tags),
        blocked_content_ids=set(user_profile.blocked_content_ids),
    )

    engine = HybridRecommendationEngine(contents, interactions, base_model, feedback_model)
    candidates = engine.generate_candidates(state, effective_profile, semantic_scores)

    # History is a candidate-level ranking signal and therefore affects final order.
    recent_history = history[-10:]
    for row in candidates:
        emotions = set(row["content"].emotions)
        matching = (
            sum(h.dominant_emotion in emotions for h in recent_history) / len(recent_history)
            if recent_history and emotions else 0.0
        )
        row["history_affinity"] = max(float(row.get("history_affinity", 0.0)), matching)

    ranked = RecommendationRanker().rank(candidates, state, top_k=top_k)
    recs = add_explanations([r.to_dict() for r in ranked], state.to_dict(), trend)

    new_history = build_history_record(user_profile.user_id, state)
    if emotion_history_csv and persist_history:
        append_emotion_history(emotion_history_csv, new_history)

    return {
        "is_valid": True,
        "input_text": text,
        "cleaned_text": processed.cleaned_text,
        "processed_text": processed.processed_text,
        "sentiment": sentiment.to_dict(),
        "emotion_model": model_type,
        "emotion": prediction.to_dict(),
        "confidence": confidence.to_dict(),
        "emotional_state": state.to_dict(),
        "emotion_history": {
            "trend": trend,
            "current_history_affinity": round(hist_affinity, 4),
            "new_record": new_history.to_dict(),
            "new_records": [new_history.to_dict()],
        },
        "semantic_similarity": semantic_scores,
        "recommendations": recs,
        "feedback": {
            "records_used_for_learning": len(feedback_records),
            "learning_enabled": bool(feedback_model and feedback_model.fitted),
            "summary": feedback_summary(feedback_records),
        },
        "personalization": {
            "base_model_fitted": bool(base_model.fitted),
            "feedback_model_fitted": bool(feedback_model and feedback_model.fitted),
            "interaction_count": len(interactions),
            "feedback_count": len(feedback_records),
        },
        "features": {
            row["content"].content_id: feature_dict(row["features"])
            for row in candidates
        },
        "candidates": [
            {
                "content_id": row["content"].content_id,
                "title": row["content"].title,
                "rule_score": row["rule_score"],
                "personalized_ml_score": row["personalized_ml_score"],
                "feedback_ml_score": row.get("feedback_ml_score", 0.0),
                "feedback_learning_active": row.get("feedback_learning_active", False),
                "history_affinity": row["history_affinity"],
                "content_similarity": row["content_similarity"],
                "preference_score": row["preference_score"],
                "collaborative_score": row["collaborative_score"],
                "emotion_relevance": row["emotion_relevance"],
                "novelty": row["novelty"],
            }
            for row in candidates
        ],
    }
