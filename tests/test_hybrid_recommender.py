from src.emotional_state import analyze_emotional_state
from src.hybrid_recommender import HybridRecommendationEngine
from src.recommendation_data import (
    DEFAULT_WELLNESS_CONTENT,
    Interaction,
    UserProfile,
)


def state():
    return analyze_emotional_state(
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


def test_rule_score_rewards_matching_emotion():
    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT)
    s = state()

    matching = next(c for c in DEFAULT_WELLNESS_CONTENT if "sadness" in c.emotions)

    assert engine._rule_score(s, matching) == 1.0


def test_preference_score_responds_to_type_and_tags():
    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT)
    profile = UserProfile(
        "u1",
        preferred_types=["breathing"],
        preferred_tags=["calm"],
    )
    content = DEFAULT_WELLNESS_CONTENT[0]

    assert engine._preference_score(profile, content) > 0


def test_blocked_content_is_not_generated():
    profile = UserProfile("u1", blocked_content_ids={"W001", "W002"})
    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT)

    rows = engine.generate_candidates(state(), profile, {})

    ids = {row["content"].content_id for row in rows}
    assert "W001" not in ids
    assert "W002" not in ids


def test_candidates_contain_all_expected_features():
    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT)
    rows = engine.generate_candidates(state(), UserProfile("u1"), {})

    assert rows

    required = {
        "content",
        "rule_score",
        "content_similarity",
        "preference_score",
        "collaborative_score",
        "emotion_relevance",
        "history_affinity",
        "novelty",
        "exposure_count",
        "personalized_ml_score",
        "features",
    }

    assert required.issubset(rows[0].keys())


def test_history_affinity_and_novelty_change_after_interaction():
    interactions = [
        Interaction("u1", "W001", 0.8),
    ]

    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT, interactions)

    history, novelty, exposure = engine._history_stats("u1", "W001")

    assert history == 0.8
    assert novelty == 0.0
    assert exposure == 1
