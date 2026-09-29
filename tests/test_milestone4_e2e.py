
def test_m4_component_end_to_end_without_network():
    from src.ingestion import ingest_raw_text
    from src.preprocessing import preprocess_text
    from src.sentiment import analyze_sentiment
    from src.emotional_state import analyze_emotional_state
    from src.hybrid_recommender import HybridRecommendationEngine
    from src.recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile
    from src.ranking import RecommendationRanker

    rec = ingest_raw_text("I feel anxious but hopeful about tomorrow.")
    assert rec.is_valid
    processed = preprocess_text(rec.raw_text)
    sentiment = analyze_sentiment(processed.cleaned_text)
    state = analyze_emotional_state(
        {"joy": 0.35, "sadness": 0.05, "anger": 0.02, "fear": 0.55, "surprise": 0.02, "disgust": 0.01},
        sentiment.compound,
    )
    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT, [], None, None)
    candidates = engine.generate_candidates(state, UserProfile("e2e", [], [], set()), {})
    ranked = RecommendationRanker().rank(candidates, state, top_k=3)
    assert ranked
    assert all(r.score >= 0 for r in ranked)
