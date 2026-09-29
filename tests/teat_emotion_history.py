import csv
import json
from types import SimpleNamespace
import pytest

from emotion_history import (
    EmotionHistoryRecord,
    append_emotion_history,
    load_emotion_history,
    analyze_emotion_trend,
    history_influence,
    build_history_record,
)


def make_record(
    user_id="user1",
    timestamp="2026-01-01T10:00:00+00:00",
    emotion="happy",
    confidence=0.9,
    intensity=0.7,
    polarity="positive",
    polarity_score=0.8,
):
    return EmotionHistoryRecord(
        user_id=user_id,
        timestamp=timestamp,
        dominant_emotion=emotion,
        confidence=confidence,
        intensity=intensity,
        polarity=polarity,
        polarity_score=polarity_score,
        triggered_emotions=["joy"],
        emotion_scores={
            "happy": 0.9,
            "sad": 0.05,
            "angry": 0.05,
        },
    )


def make_emotional_state():
    return SimpleNamespace(
        dominant_emotion="happy",
        emotion_confidence=0.92,
        intensity=0.75,
        polarity="positive",
        polarity_score=0.85,
        triggered_emotions=["joy", "calm"],
        emotion_probabilities={
            "happy": 0.92,
            "sad": 0.03,
            "angry": 0.05,
        },
    )


# ---------------------------------------------------------
# EmotionHistoryRecord
# ---------------------------------------------------------

def test_record_to_dict():
    record = make_record()

    result = record.to_dict()

    assert result["user_id"] == "user1"
    assert result["dominant_emotion"] == "happy"
    assert result["confidence"] == 0.9
    assert result["intensity"] == 0.7
    assert result["triggered_emotions"] == ["joy"]
    assert result["emotion_scores"]["happy"] == 0.9


# ---------------------------------------------------------
# append_emotion_history / load_emotion_history
# ---------------------------------------------------------

def test_append_and_load_history(tmp_path):
    path = tmp_path / "history.csv"

    record = make_record()

    append_emotion_history(str(path), record)

    assert path.exists()

    loaded = load_emotion_history(str(path))

    assert len(loaded) == 1

    result = loaded[0]

    assert result.user_id == "user1"
    assert result.timestamp == record.timestamp
    assert result.dominant_emotion == "happy"
    assert result.confidence == 0.9
    assert result.intensity == 0.7
    assert result.polarity == "positive"
    assert result.polarity_score == 0.8
    assert result.triggered_emotions == ["joy"]
    assert result.emotion_scores["happy"] == 0.9


def test_append_multiple_records(tmp_path):
    path = tmp_path / "history.csv"

    record1 = make_record(
        timestamp="2026-01-01T10:00:00+00:00",
        emotion="happy",
    )

    record2 = make_record(
        timestamp="2026-01-01T11:00:00+00:00",
        emotion="sad",
        polarity="negative",
        polarity_score=-0.8,
    )

    append_emotion_history(str(path), record1)
    append_emotion_history(str(path), record2)

    loaded = load_emotion_history(str(path))

    assert len(loaded) == 2
    assert loaded[0].dominant_emotion == "happy"
    assert loaded[1].dominant_emotion == "sad"


def test_load_history_filters_by_user(tmp_path):
    path = tmp_path / "history.csv"

    append_emotion_history(
        str(path),
        make_record(user_id="alice"),
    )

    append_emotion_history(
        str(path),
        make_record(
            user_id="bob",
            timestamp="2026-01-01T11:00:00+00:00",
        ),
    )

    loaded = load_emotion_history(
        str(path),
        user_id="alice",
    )

    assert len(loaded) == 1
    assert loaded[0].user_id == "alice"


def test_load_missing_file_returns_empty_list(tmp_path):
    path = tmp_path / "does_not_exist.csv"

    result = load_emotion_history(str(path))

    assert result == []


def test_csv_contains_expected_headers(tmp_path):
    path = tmp_path / "history.csv"

    append_emotion_history(
        str(path),
        make_record(),
    )

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        assert reader.fieldnames == [
            "user_id",
            "timestamp",
            "dominant_emotion",
            "confidence",
            "intensity",
            "polarity",
            "polarity_score",
            "triggered_emotions",
            "emotion_scores",
        ]


# ---------------------------------------------------------
# analyze_emotion_trend
# ---------------------------------------------------------

def test_analyze_empty_history():
    result = analyze_emotion_trend([])

    assert result["sample_count"] == 0
    assert result["emotion_frequency"] == {}
    assert result["dominant_emotions"] == []
    assert result["average_intensity"] == 0.0
    assert result["recent_average_intensity"] == 0.0
    assert result["intensity_delta"] == 0.0
    assert result["positive_count"] == 0
    assert result["negative_count"] == 0
    assert result["polarity_trend"] == "no_data"
    assert result["repeated_emotions"] == []
    assert result["recent_state"] is None


def test_analyze_emotion_frequency():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            emotion="happy",
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            emotion="sad",
            polarity="negative",
            polarity_score=-0.7,
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            emotion="happy",
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["sample_count"] == 3
    assert result["emotion_frequency"]["happy"] == 2
    assert result["emotion_frequency"]["sad"] == 1
    assert result["dominant_emotions"] == ["happy"]


def test_analyze_average_intensity():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            intensity=0.2,
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            intensity=0.6,
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            intensity=1.0,
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["average_intensity"] == 0.6


def test_recent_average_and_delta():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            intensity=0.2,
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            intensity=0.4,
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            intensity=0.8,
        ),
        make_record(
            timestamp="2026-01-01T13:00:00+00:00",
            intensity=1.0,
        ),
    ]

    result = analyze_emotion_trend(
        records,
        recent_n=2,
    )

    assert result["recent_average_intensity"] == 0.9
    assert result["intensity_delta"] == 0.4


def test_polarity_counts():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            polarity="positive",
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            polarity="negative",
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            polarity="positive",
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["positive_count"] == 2
    assert result["negative_count"] == 1


def test_polarity_trend_improving():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            polarity_score=-0.8,
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            polarity_score=-0.6,
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            polarity_score=0.5,
        ),
        make_record(
            timestamp="2026-01-01T13:00:00+00:00",
            polarity_score=0.8,
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["polarity_trend"] == "improving"


def test_polarity_trend_worsening():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            polarity_score=0.8,
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            polarity_score=0.6,
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            polarity_score=-0.5,
        ),
        make_record(
            timestamp="2026-01-01T13:00:00+00:00",
            polarity_score=-0.8,
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["polarity_trend"] == "worsening"


def test_polarity_trend_stable():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            polarity_score=0.5,
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            polarity_score=0.52,
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            polarity_score=0.51,
        ),
        make_record(
            timestamp="2026-01-01T13:00:00+00:00",
            polarity_score=0.53,
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["polarity_trend"] == "stable"


def test_single_record_has_insufficient_polarity_data():
    records = [make_record()]

    result = analyze_emotion_trend(records)

    assert result["polarity_trend"] == "insufficient_data"


def test_repeated_emotions():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            emotion="happy",
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            emotion="happy",
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            emotion="sad",
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["repeated_emotions"] == ["happy"]


def test_recent_state_is_last_record():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            emotion="sad",
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            emotion="happy",
        ),
    ]

    result = analyze_emotion_trend(records)

    assert result["recent_state"]["dominant_emotion"] == "happy"


# ---------------------------------------------------------
# history_influence
# ---------------------------------------------------------

def test_history_influence_no_records():
    result = history_influence([], "happy")

    assert result == 0.0


def test_history_influence_all_match():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            emotion="happy",
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            emotion="happy",
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            emotion="happy",
        ),
    ]

    result = history_influence(
        records,
        "happy",
    )

    assert result == 1.0


def test_history_influence_partial_match():
    records = [
        make_record(
            timestamp="2026-01-01T10:00:00+00:00",
            emotion="happy",
        ),
        make_record(
            timestamp="2026-01-01T11:00:00+00:00",
            emotion="sad",
        ),
        make_record(
            timestamp="2026-01-01T12:00:00+00:00",
            emotion="happy",
        ),
        make_record(
            timestamp="2026-01-01T13:00:00+00:00",
            emotion="angry",
        ),
    ]

    result = history_influence(
        records,
        "happy",
    )

    assert result == 0.5


def test_history_influence_only_uses_last_10_records():
    records = []

    for i in range(11):
        records.append(
            make_record(
                timestamp=f"2026-01-01T{i:02d}:00:00+00:00",
                emotion="happy" if i == 0 else "sad",
            )
        )

    result = history_influence(
        records,
        "happy",
    )

    assert result == 0.0


# ---------------------------------------------------------
# build_history_record
# ---------------------------------------------------------

def test_build_history_record():
    emotional_state = make_emotional_state()

    record = build_history_record(
        user_id="user123",
        emotional_state=emotional_state,
    )

    assert record.user_id == "user123"
    assert record.dominant_emotion == "happy"
    assert record.confidence == 0.92
    assert record.intensity == 0.75
    assert record.polarity == "positive"
    assert record.polarity_score == 0.85

    assert record.triggered_emotions == [
        "joy",
        "calm",
    ]

    assert record.emotion_scores == {
        "happy": 0.92,
        "sad": 0.03,
        "angry": 0.05,
    }

    # ISO timestamp should be parseable.
    datetime_value = record.timestamp

    from datetime import datetime

    parsed = datetime.fromisoformat(datetime_value)

    assert parsed.tzinfo is not None


# ---------------------------------------------------------
# JSON serialization inside CSV
# ---------------------------------------------------------

def test_json_fields_are_serialized(tmp_path):
    path = tmp_path / "history.csv"

    record = make_record()

    append_emotion_history(
        str(path),
        record,
    )

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        row = next(csv.DictReader(f))

    triggered = json.loads(
        row["triggered_emotions"]
    )

    scores = json.loads(
        row["emotion_scores"]
    )

    assert triggered == ["joy"]
    assert scores["happy"] == 0.9
