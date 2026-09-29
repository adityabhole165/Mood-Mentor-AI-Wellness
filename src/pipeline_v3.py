"""Milestone 3 end-to-end pipeline. Milestones 1 and 2 remain unchanged."""
from __future__ import annotations
import os
from .ingestion import read_input_data
from .preprocessing import preprocess_text
from .sentiment import analyze_sentiment
from . import emotion_bert
from .confidence import build_confidence_report
from .emotional_state import analyze_emotional_state
from .recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile
from .data_loader import load_wellness_content, load_interactions
from .semantic_matching import SemanticMatcher
from .hybrid_recommender import HybridRecommendationEngine
from .ranking import RecommendationRanker
from .personalized_recommender import train_personalized_model
from .recommendation_feedback import load_feedback, train_model_from_feedback
from .emotion_history import load_emotion_history, append_emotion_history, analyze_emotion_trend, build_history_record, history_influence
from .recommendation_explainability import add_explanations

def run_milestone3(text, user_profile, interactions=None, contents=None, emotion_model_dir=None,
                   wellness_content_csv=None, interactions_csv=None, top_k=5, semantic_model_name="all-MiniLM-L6-v2",
                   emotion_history_csv=None, feedback_csv=None, persist_history=True):
    if wellness_content_csv and os.path.exists(wellness_content_csv): contents = load_wellness_content(wellness_content_csv)
    contents = contents or DEFAULT_WELLNESS_CONTENT
    if interactions is None and interactions_csv: interactions = load_interactions(interactions_csv)
    interactions = interactions or []
    emotion_model_dir = emotion_model_dir or emotion_bert.DEFAULT_SAVE_DIR

    records = read_input_data("raw_text", text)
    record = records[0]
    if not record.is_valid:
        return {"is_valid": False, "error": record.error, "recommendations": []}

    processed = preprocess_text(record.raw_text)
    sentiment = analyze_sentiment(processed.cleaned_text)
    model, tokenizer = emotion_bert.load_trained_model(emotion_model_dir)
    prediction = emotion_bert.predict(processed.cleaned_text, model, tokenizer)
    confidence = build_confidence_report(prediction.scores)
    state = analyze_emotional_state(confidence.all_scores, sentiment.compound)

    history = load_emotion_history(emotion_history_csv, user_profile.user_id) if emotion_history_csv else []
    trend = analyze_emotion_trend(history)
    hist_affinity = history_influence(history, state.dominant_emotion)

    matcher = SemanticMatcher(semantic_model_name)
    semantic_scores = matcher.match_emotional_state(state, record.raw_text, contents)
    personalized_model = train_personalized_model(interactions)
    feedback_records = load_feedback(feedback_csv, user_profile.user_id) if feedback_csv else []
    feedback_model = train_model_from_feedback(feedback_records) if feedback_records else None
    engine = HybridRecommendationEngine(contents, interactions, personalized_model, feedback_model)
    candidates = engine.generate_candidates(state, user_profile, semantic_scores)
    recent_history = history[-10:]
    for row in candidates:
        if recent_history and row["content"].emotions:
            matching = sum(h.dominant_emotion in set(row["content"].emotions) for h in recent_history) / len(recent_history)
            row["history_affinity"] = max(float(row.get("history_affinity", 0.0)), matching)
    ranked = RecommendationRanker().rank(candidates, state, top_k=top_k)
    if emotion_history_csv and persist_history:
        append_emotion_history(emotion_history_csv, build_history_record(user_profile.user_id, state))

    return {"is_valid": True, "input_text": text, "sentiment": sentiment.to_dict(), "emotion": prediction.to_dict(),
            "emotional_state": state.to_dict(), "semantic_similarity": semantic_scores,
            "recommendations": add_explanations([r.to_dict() for r in ranked], state.to_dict(), trend),
            "emotion_history": {"trend": trend, "current_history_affinity": round(hist_affinity, 4)},
            "feedback": {"records_used_for_learning": len(feedback_records), "learning_enabled": bool(feedback_model and feedback_model.fitted)}}
