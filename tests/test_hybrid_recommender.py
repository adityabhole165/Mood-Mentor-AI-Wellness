from src.emotional_state import analyze_emotional_state
from src.hybrid_recommender import HybridRecommendationEngine
from src.ranking import RecommendationRanker
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


def test_feedback_model_is_blended_into_hybrid_score():
    from src.personalized_recommender import PersonalizedModel
    state = analyze_emotional_state({"joy": .05, "sadness": .6, "anger": .1, "fear": .7, "surprise": .05, "disgust": .02}, -.6)
    profile = UserProfile("u1")
    base = PersonalizedModel(baseline=0.2)
    feedback = PersonalizedModel(baseline=0.9, fitted=True)
    rows = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT, [], base, feedback).generate_candidates(state, profile, {})
    assert rows
    assert all(r["feedback_learning_active"] for r in rows)
    assert all(abs(r["personalized_ml_score"] - (.65*.2 + .35*.9)) < 1e-9 for r in rows)


def test_history_affinity_can_change_rank_order():
    from src.emotion_history import EmotionHistoryRecord
    state = analyze_emotional_state({"joy": .02, "sadness": .1, "anger": .05, "fear": .9, "surprise": .02, "disgust": .01}, -.8)
    profile = UserProfile("u1")
    rows = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT, []).generate_candidates(state, profile, {c.content_id: 0.1 for c in DEFAULT_WELLNESS_CONTENT})
    ranker = RecommendationRanker(low_relevance_threshold=0.0)
    before = [x.content_id for x in ranker.rank(rows, state, 7)]
    for row in rows:
        if row["content"].content_id == "W007":
            row["history_affinity"] = 1.0
    after = [x.content_id for x in ranker.rank(rows, state, 7)]
    assert before != after
