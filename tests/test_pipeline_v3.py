import pytest

import src.pipeline_v3 as pipeline_v3

from src.emotion_bert import EmotionPrediction
from src.recommendation_data import UserProfile, DEFAULT_WELLNESS_CONTENT


def test_invalid_input_returns_safe_result(monkeypatch):
    result = pipeline_v3.run_milestone3(
        "",
        UserProfile("u1"),
        contents=DEFAULT_WELLNESS_CONTENT,
        emotion_model_dir="unused",
    )

    assert result["is_valid"] is False
    assert result["recommendations"] == []
    assert "error" in result


def test_end_to_end_pipeline_with_mocked_emotion_model(monkeypatch):
    # Avoid loading a real transformer during unit testing.
    scores = {
        "joy": 0.02,
        "sadness": 0.75,
        "anger": 0.05,
        "fear": 0.65,
        "surprise": 0.02,
        "disgust": 0.01,
    }

    prediction = EmotionPrediction(
        text="I feel sad and afraid",
        primary_emotion="sadness",
        primary_confidence=0.80,
        scores=scores,
        triggered_emotions=["sadness", "fear"],
        threshold=0.35,
    )

    monkeypatch.setattr(
        pipeline_v3.emotion_bert,
        "load_trained_model",
        lambda model_dir: (object(), object()),
    )

    monkeypatch.setattr(
        pipeline_v3.emotion_bert,
        "predict",
        lambda text, model, tokenizer: prediction,
    )

    # Avoid downloading all-MiniLM during this test.
    class FakeMatcher:
        def __init__(self, model_name):
            self.model_name = model_name

        def match_emotional_state(self, state, original_text, contents):
            return {c.content_id: 0.5 for c in contents}

    monkeypatch.setattr(
        pipeline_v3,
        "SemanticMatcher",
        FakeMatcher,
    )

    result = pipeline_v3.run_milestone3(
        "I feel anxious and overwhelmed today.",
        UserProfile(
            "u1",
            preferred_types=["breathing"],
            preferred_tags=["calm", "stress"],
        ),
        interactions=[],
        contents=DEFAULT_WELLNESS_CONTENT,
        emotion_model_dir="unused",
        top_k=3,
    )

    assert result["is_valid"] is True
    assert result["input_text"].startswith("I feel")
    assert result["sentiment"]
    assert result["emotion"]["primary_emotion"] == "sadness"
    assert result["emotional_state"]["dominant_emotion"] == "sadness"

    assert set(result["semantic_similarity"]) == {
        c.content_id for c in DEFAULT_WELLNESS_CONTENT
    }

    assert len(result["recommendations"]) <= 3

    if result["recommendations"]:
        ranks = [r["rank"] for r in result["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))


def test_pipeline_respects_blocked_content(monkeypatch):
    scores = {
        "joy": 0.01,
        "sadness": 0.80,
        "anger": 0.02,
        "fear": 0.70,
        "surprise": 0.01,
        "disgust": 0.01,
    }

    # IMPORTANT:
    # EmotionPrediction now requires the text field.
    # Use keyword arguments to avoid positional-argument mistakes.
    prediction = EmotionPrediction(
        text="I feel stressed.",
        primary_emotion="sadness",
        primary_confidence=0.80,
        scores=scores,
        triggered_emotions=["sadness", "fear"],
        threshold=0.35,
    )

    monkeypatch.setattr(
        pipeline_v3.emotion_bert,
        "load_trained_model",
        lambda model_dir: (object(), object()),
    )

    monkeypatch.setattr(
        pipeline_v3.emotion_bert,
        "predict",
        lambda text, model, tokenizer: prediction,
    )

    class FakeMatcher:
        def __init__(self, model_name):
            pass

        def match_emotional_state(self, state, original_text, contents):
            return {c.content_id: 0.9 for c in contents}

    monkeypatch.setattr(
        pipeline_v3,
        "SemanticMatcher",
        FakeMatcher,
    )

    blocked = DEFAULT_WELLNESS_CONTENT[0].content_id

    result = pipeline_v3.run_milestone3(
        "I feel stressed.",
        UserProfile(
            "u1",
            blocked_content_ids={blocked},
        ),
        interactions=[],
        contents=DEFAULT_WELLNESS_CONTENT,
        emotion_model_dir="unused",
        top_k=7,
    )

    assert all(
        r["content_id"] != blocked
        for r in result["recommendations"]
    )

