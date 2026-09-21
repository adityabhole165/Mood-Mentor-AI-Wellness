"""Smoke tests tying the individual Milestone 3 modules together."""

from src.emotional_state import analyze_emotional_state
from src.recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile
from src.hybrid_recommender import HybridRecommendationEngine
from src.ranking import RecommendationRanker


def test_complete_non_transformer_recommendation_stack():
    state = analyze_emotional_state(
        {
            "joy": 0.01,
            "sadness": 0.70,
            "anger": 0.02,
            "fear": 0.65,
            "surprise": 0.01,
            "disgust": 0.01,
        },
        -0.5,
    )

    profile = UserProfile(
        "u1",
        preferred_types=["breathing"],
        preferred_tags=["calm", "grounding"],
    )

    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT, [])
    candidates = engine.generate_candidates(
        state,
        profile,
        semantic_scores={c.content_id: 0.5 for c in DEFAULT_WELLNESS_CONTENT},
    )

    ranked = RecommendationRanker().rank(candidates, state, top_k=5)

    assert candidates
    assert len(ranked) <= 5
    assert len({r.content_id for r in ranked}) == len(ranked)
    assert all(0 <= r.score <= 1 for r in ranked)
