import pytest

import src.semantic_matching as sm
from src.emotional_state import analyze_emotional_state
from src.recommendation_data import DEFAULT_WELLNESS_CONTENT


def make_state():
    return analyze_emotional_state(
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


def test_tfidf_fallback_scores_every_content(monkeypatch):
    # Prevent a transformer download in the unit test.
    monkeypatch.setattr(sm, "SentenceTransformer", None)

    matcher = sm.SemanticMatcher(allow_tfidf_fallback=True)
    scores = matcher.match_emotional_state(
        make_state(),
        "I feel anxious and overwhelmed",
        DEFAULT_WELLNESS_CONTENT,
    )

    assert set(scores) == {c.content_id for c in DEFAULT_WELLNESS_CONTENT}
    assert all(isinstance(v, float) for v in scores.values())


def test_cosine_zero_vector_is_zero():
    assert sm.SemanticMatcher._cosine([0, 0], [1, 2]) == 0.0


def test_semantic_scores_are_bounded_for_tfidf(monkeypatch):
    monkeypatch.setattr(sm, "SentenceTransformer", None)

    matcher = sm.SemanticMatcher(allow_tfidf_fallback=True)
    scores = matcher.score("calm breathing anxiety", DEFAULT_WELLNESS_CONTENT)

    assert all(-1e-9 <= v <= 1.0000001 for v in scores.values())
