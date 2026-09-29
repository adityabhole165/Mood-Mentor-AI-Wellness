from pathlib import Path

from src.emotional_state import analyze_emotional_state
from src.emotion_history import (
    EmotionHistoryRecord,
    append_emotion_history,
    load_emotion_history,
    analyze_emotion_trend,
)
from src.recommendation_feedback import (
    record_feedback,
    load_feedback,
    train_model_from_feedback,
)
from src.recommendation_explainability import (
    explain_recommendation,
)


FEATURES = {
    "emotion_relevance": 0.8,
    "emotion_intensity": 0.7,
    "user_preference": 0.6,
    "content_similarity": 0.7,
    "collaborative_score": 0.3,
    "history_affinity": 0.4,
    "novelty": 0.5,
    "negative_severity": 0.6,
}


def test_emotional_state_intensity_is_valid():
    state = analyze_emotional_state(
        {
            "joy": 0.72,
            "sadness": 0.12,
            "anger": 0.03,
            "fear": 0.20,
            "surprise": 0.10,
            "disgust": 0.01,
        },
        0.55,
    )

    assert 0 <= state.intensity <= 1


def test_emotion_history_round_trip(tmp_path):
    history_file = Path(tmp_path) / "emotion_history.csv"

    append_emotion_history(
        str(history_file),
        EmotionHistoryRecord(
            "u1",
            "2026-01-01T00:00:00+00:00",
            "fear",
            0.8,
            0.7,
            "negative",
            -0.6,
            ["fear"],
            {"fear": 0.8},
        ),
    )

    append_emotion_history(
        str(history_file),
        EmotionHistoryRecord(
            "u1",
            "2026-01-02T00:00:00+00:00",
            "fear",
            0.7,
            0.6,
            "negative",
            -0.4,
            ["fear"],
            {"fear": 0.7},
        ),
    )

    history = load_emotion_history(
        str(history_file),
        "u1",
    )

    assert len(history) == 2

    trend = analyze_emotion_trend(history)

    assert trend["sample_count"] == 2
    assert "fear" in trend["repeated_emotions"]


def test_feedback_learning_pipeline(tmp_path):
    feedback_file = Path(tmp_path) / "feedback.csv"

    record_feedback(
        str(feedback_file),
        "u1",
        "W001",
        "accept",
        rating=5,
        feature_vector=FEATURES,
    )

    record_feedback(
        str(feedback_file),
        "u1",
        "W002",
        "reject",
        rating=1,
        feature_vector=FEATURES,
    )

    feedback = load_feedback(
        str(feedback_file),
        "u1",
    )

    assert len(feedback) == 2

    model = train_model_from_feedback(feedback)

    assert model is not None


def test_dynamic_recommendation_explanation():
    state = {
        "dominant_emotion": "fear",
        "intensity": 0.70,
    }

    trend = {
        "sample_count": 2,
        "repeated_emotions": ["fear"],
        "polarity_trend": "worsening",
    }

    recommendation = {
        "content_id": "W001",
        "title": "Breathing",
        "score": 0.80,
        "components": {
            "emotion_relevance": 0.80,
            "emotion_intensity": 0.70,
            "user_preference": 0.60,
            "content_similarity": 0.70,
            "previous_interaction": 0.40,
        },
    }

    explanation = explain_recommendation(
        recommendation,
        state,
        trend,
    )

    assert explanation["content_id"] == "W001"
    assert explanation["title"] == "Breathing"
    assert explanation["score"] == 0.80
    assert explanation["reasons"]
    assert explanation["evidence"]["detected_emotion"] == "fear"
    assert explanation["evidence"]["emotion_intensity"] == 0.70


def test_task10_complete_regression_flow(tmp_path):
    history_file = Path(tmp_path) / "emotion_history.csv"
    feedback_file = Path(tmp_path) / "feedback.csv"

    # 1. Create emotional history
    append_emotion_history(
        str(history_file),
        EmotionHistoryRecord(
            "u1",
            "2026-01-01T00:00:00+00:00",
            "fear",
            0.8,
            0.7,
            "negative",
            -0.6,
            ["fear"],
            {"fear": 0.8},
        ),
    )

    history = load_emotion_history(
        str(history_file),
        "u1",
    )

    trend = analyze_emotion_trend(history)

    assert trend["sample_count"] == 1

    # 2. Record recommendation feedback
    record_feedback(
        str(feedback_file),
        "u1",
        "W001",
        "accept",
        rating=5,
        feature_vector=FEATURES,
    )

    record_feedback(
        str(feedback_file),
        "u1",
        "W002",
        "reject",
        rating=1,
        feature_vector=FEATURES,
    )

    feedback = load_feedback(
        str(feedback_file),
        "u1",
    )

    assert len(feedback) == 2

    # 3. Train model
    model = train_model_from_feedback(feedback)

    assert model is not None

    # 4. Generate explanation
    explanation = explain_recommendation(
        {
            "content_id": "W001",
            "title": "Breathing",
            "score": 0.8,
            "components": {
                "emotion_relevance": 0.8,
                "emotion_intensity": 0.7,
                "user_preference": 0.6,
                "content_similarity": 0.7,
                "previous_interaction": 0.4,
            },
        },
        {
            "dominant_emotion": "fear",
            "intensity": 0.7,
        },
        trend,
    )

    assert explanation["reasons"]