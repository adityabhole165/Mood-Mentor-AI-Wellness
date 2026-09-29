import json
from pathlib import Path

import pytest

from src.recommendation_feedback import (
    VALID_EVENTS,
    FeedbackRecord,
    _reward,
    record_feedback,
    load_feedback,
    feedback_summary,
    train_model_from_feedback,
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


def test_all_feedback_events_are_supported():
    expected = {
        "view",
        "accept",
        "reject",
        "skip",
        "complete",
        "helpful",
        "hide",
        "rate",
    }

    assert VALID_EVENTS == expected


@pytest.mark.parametrize(
    "event, expected",
    [
        ("view", 0.10),
        ("accept", 0.80),
        ("complete", 0.90),
        ("helpful", 1.00),
        ("reject", 0.00),
        ("skip", 0.00),
        ("hide", 0.00),
    ],
)
def test_reward_for_events(event, expected):
    assert _reward(event, None) == expected


@pytest.mark.parametrize(
    "rating, expected",
    [
        (0, 0.0),
        (1, 0.2),
        (2.5, 0.5),
        (5, 1.0),
        (10, 1.0),
        (-5, 0.0),
    ],
)
def test_rating_reward_is_normalized_and_clamped(rating, expected):
    assert _reward("rate", rating) == expected


def test_record_feedback_creates_csv(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

    record = record_feedback(
        str(feedback_file),
        "u1",
        "W001",
        "accept",
        rating=5,
        feature_vector=FEATURES,
        preference_changes={"preferred_type": "breathing"},
        recommendation_score=0.87,
    )

    assert isinstance(record, FeedbackRecord)
    assert record.user_id == "u1"
    assert record.content_id == "W001"
    assert record.event == "accept"
    assert record.rating == 5
    assert record.recommendation_score == 0.87
    assert feedback_file.exists()


def test_record_feedback_writes_header_only_once(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

    record_feedback(
        str(feedback_file),
        "u1",
        "W001",
        "accept",
        feature_vector=FEATURES,
    )

    record_feedback(
        str(feedback_file),
        "u1",
        "W002",
        "reject",
        feature_vector=FEATURES,
    )

    lines = feedback_file.read_text(encoding="utf-8").splitlines()

    # Header + two records
    assert len(lines) == 3


def test_invalid_event_raises_value_error(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

    with pytest.raises(ValueError):
        record_feedback(
            str(feedback_file),
            "u1",
            "W001",
            "invalid_event",
        )


def test_load_feedback_filters_by_user(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

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
        "u2",
        "W002",
        "reject",
        rating=1,
        feature_vector=FEATURES,
    )

    all_records = load_feedback(str(feedback_file))
    user_records = load_feedback(str(feedback_file), "u1")

    assert len(all_records) == 2
    assert len(user_records) == 1
    assert user_records[0].user_id == "u1"


def test_load_feedback_missing_file_returns_empty_list(tmp_path):
    feedback_file = tmp_path / "missing.csv"

    records = load_feedback(str(feedback_file))

    assert records == []


def test_feedback_summary(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

    record_feedback(
        str(feedback_file),
        "u1",
        "W001",
        "view",
        feature_vector=FEATURES,
    )

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

    records = load_feedback(str(feedback_file), "u1")
    summary = feedback_summary(records)

    assert summary["total"] == 3
    assert summary["event_counts"]["view"] == 1
    assert summary["event_counts"]["accept"] == 1
    assert summary["event_counts"]["reject"] == 1
    assert summary["average_rating"] == 3.0
    assert summary["acceptance_rate"] == 1.0


def test_feedback_json_fields_are_serialized(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

    record_feedback(
        str(feedback_file),
        "u1",
        "W001",
        "accept",
        preference_changes={"content_type": "meditation"},
        feature_vector=FEATURES,
    )

    records = load_feedback(str(feedback_file), "u1")

    assert records[0].preference_changes["content_type"] == "meditation"
    assert records[0].feature_vector["emotion_relevance"] == 0.8


def test_train_model_from_feedback(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

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

    records = load_feedback(str(feedback_file), "u1")

    model = train_model_from_feedback(records)

    assert model is not None