"""Task 10 lightweight regression checks for the Milestone 3 Part 2 modules."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .emotional_state import analyze_emotional_state
from .emotion_history import (
    EmotionHistoryRecord, append_emotion_history,
    load_emotion_history, analyze_emotion_trend
)
from .recommendation_feedback import record_feedback, load_feedback, train_model_from_feedback
from .recommendation_explainability import explain_recommendation


def run_regression_tests() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        history_file = root / "emotion_history.csv"
        feedback_file = root / "feedback.csv"

        state = analyze_emotional_state(
            {"joy": .72, "sadness": .12, "anger": .03, "fear": .20, "surprise": .10, "disgust": .01},
            0.55,
        )
        assert 0 <= state.intensity <= 1

        append_emotion_history(
            str(history_file),
            EmotionHistoryRecord(
                "u1", "2026-01-01T00:00:00+00:00", "fear", .8, .7,
                "negative", -.6, ["fear"], {"fear": .8}
            ),
        )
        append_emotion_history(
            str(history_file),
            EmotionHistoryRecord(
                "u1", "2026-01-02T00:00:00+00:00", "fear", .7, .6,
                "negative", -.4, ["fear"], {"fear": .7}
            ),
        )
        history = load_emotion_history(str(history_file), "u1")
        trend = analyze_emotion_trend(history)
        assert trend["sample_count"] == 2
        assert "fear" in trend["repeated_emotions"]

        features = {
            "emotion_relevance": .8, "emotion_intensity": .7,
            "user_preference": .6, "content_similarity": .7,
            "collaborative_score": .3, "history_affinity": .4,
            "novelty": .5, "negative_severity": .6,
        }
        record_feedback(
            str(feedback_file), "u1", "W001", "accept",
            feature_vector=features, rating=5
        )
        record_feedback(
            str(feedback_file), "u1", "W002", "reject",
            feature_vector=features, rating=1
        )
        feedback = load_feedback(str(feedback_file), "u1")
        model = train_model_from_feedback(feedback)
        assert model is not None

        explanation = explain_recommendation(
            {
                "content_id": "W001", "title": "Breathing",
                "score": .8,
                "components": {
                    "emotion_relevance": .8, "emotional_intensity": .7,
                    "user_preference": .6, "content_similarity": .7,
                    "previous_interaction": .4,
                }
            },
            state.to_dict(),
            trend,
        )
        assert explanation["reasons"]

    return {"passed": True, "checks": 8}


if __name__ == "__main__":
    print(run_regression_tests())
