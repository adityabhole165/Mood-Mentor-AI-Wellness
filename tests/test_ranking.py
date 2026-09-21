from src.emotional_state import analyze_emotional_state
from src.ranking import RecommendationRanker
from src.recommendation_data import DEFAULT_WELLNESS_CONTENT


def make_state(intensity=0.7):
    state = analyze_emotional_state(
        {
            "joy": 0.01,
            "sadness": 0.80,
            "anger": 0.02,
            "fear": 0.70,
            "surprise": 0.01,
            "disgust": 0.01,
        },
        -0.6,
    )
    state.intensity = intensity
    return state


def row(content, score=0.8):
    return {
        "content": content,
        "personalized_ml_score": score,
        "emotion_relevance": score,
        "content_similarity": score,
        "preference_score": score,
        "collaborative_score": score,
        "history_affinity": score,
        "rule_score": score,
        "novelty": score,
    }


def test_weights_sum_to_one():
    ranker = RecommendationRanker()
    weights = ranker._weights(0.7)

    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_score_row_is_bounded():
    ranker = RecommendationRanker()
    score = ranker.score_row(row(DEFAULT_WELLNESS_CONTENT[0]), make_state())

    assert 0 <= score <= 1


def test_rank_is_sorted_and_top_k_applied():
    ranker = RecommendationRanker(low_relevance_threshold=0.0)
    rows = [
        row(DEFAULT_WELLNESS_CONTENT[0], 0.9),
        row(DEFAULT_WELLNESS_CONTENT[1], 0.4),
        row(DEFAULT_WELLNESS_CONTENT[2], 0.7),
    ]

    results = ranker.rank(rows, make_state(), top_k=2)

    assert len(results) == 2
    assert results == sorted(results, key=lambda x: (-x.score, x.content_id))
    assert [x.rank for x in results] == [1, 2]


def test_rank_deduplicates_content_ids():
    ranker = RecommendationRanker(low_relevance_threshold=0.0)
    content = DEFAULT_WELLNESS_CONTENT[0]

    low = row(content, 0.2)
    high = row(content, 0.9)

    results = ranker.rank([low, high], make_state(), top_k=5)

    assert len(results) == 1
    assert results[0].score > 0.2


def test_low_relevance_items_are_filtered():
    ranker = RecommendationRanker(low_relevance_threshold=0.9)

    low = row(DEFAULT_WELLNESS_CONTENT[0], 0.01)
    results = ranker.rank([low], make_state(), top_k=5)

    assert results == []
