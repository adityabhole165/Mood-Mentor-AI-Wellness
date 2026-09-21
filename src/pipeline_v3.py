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

def run_milestone3(text, user_profile, interactions=None, contents=None, emotion_model_dir=None,
                   wellness_content_csv=None, interactions_csv=None, top_k=5, semantic_model_name="all-MiniLM-L6-v2"):
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

    matcher = SemanticMatcher(semantic_model_name)
    semantic_scores = matcher.match_emotional_state(state, record.raw_text, contents)
    personalized_model = train_personalized_model(interactions)
    engine = HybridRecommendationEngine(contents, interactions, personalized_model)
    candidates = engine.generate_candidates(state, user_profile, semantic_scores)
    ranked = RecommendationRanker().rank(candidates, state, top_k=top_k)

    return {"is_valid": True, "input_text": text, "sentiment": sentiment.to_dict(), "emotion": prediction.to_dict(),
            "emotional_state": state.to_dict(), "semantic_similarity": semantic_scores,
            "recommendations": [r.to_dict() for r in ranked]}
