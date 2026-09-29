from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List


@dataclass
class EmotionHistoryRecord:
    user_id: str
    timestamp: str
    dominant_emotion: str
    confidence: float
    intensity: float
    polarity: str
    polarity_score: float
    triggered_emotions: List[str]
    emotion_scores: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


FIELDS = [
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


def append_emotion_history(
    path: str,
    record: EmotionHistoryRecord
) -> None:

    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    exists = (
        os.path.exists(path)
        and os.path.getsize(path) > 0
    )

    with open(
        path,
        "a",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            "user_id": record.user_id,
            "timestamp": record.timestamp,
            "dominant_emotion": record.dominant_emotion,
            "confidence": record.confidence,
            "intensity": record.intensity,
            "polarity": record.polarity,
            "polarity_score": record.polarity_score,
            "triggered_emotions": json.dumps(
                record.triggered_emotions
            ),
            "emotion_scores": json.dumps(
                record.emotion_scores
            ),
        })


def load_emotion_history(
    path: str,
    user_id: str | None = None
) -> List[EmotionHistoryRecord]:

    import json

    if not os.path.exists(path):
        return []

    result = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        for row in csv.DictReader(f):

            if (
                user_id
                and row.get("user_id", "").strip()
                != user_id
            ):
                continue

            try:
                triggered = json.loads(
                    row.get("triggered_emotions") or "[]"
                )

                scores = json.loads(
                    row.get("emotion_scores") or "{}"
                )

            except json.JSONDecodeError:
                triggered = []
                scores = {}

            result.append(
                EmotionHistoryRecord(
                    user_id=row.get(
                        "user_id", ""
                    ).strip(),

                    timestamp=row.get(
                        "timestamp", ""
                    ).strip(),

                    dominant_emotion=row.get(
                        "dominant_emotion", ""
                    ).strip(),

                    confidence=float(
                        row.get("confidence") or 0
                    ),

                    intensity=float(
                        row.get("intensity") or 0
                    ),

                    polarity=row.get(
                        "polarity",
                        "neutral"
                    ).strip(),

                    polarity_score=float(
                        row.get("polarity_score") or 0
                    ),

                    triggered_emotions=triggered,

                    emotion_scores={
                        str(k): float(v)
                        for k, v in scores.items()
                    },
                )
            )

    return result


def _sorted(
    records: List[EmotionHistoryRecord]
) -> List[EmotionHistoryRecord]:

    return sorted(
        records,
        key=lambda x: x.timestamp
    )


def analyze_emotion_trend(
    records: List[EmotionHistoryRecord],
    recent_n: int = 5
) -> dict:

    if not records:
        return {
            "sample_count": 0,
            "emotion_frequency": {},
            "dominant_emotions": [],
            "average_intensity": 0.0,
            "recent_average_intensity": 0.0,
            "intensity_delta": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "polarity_trend": "no_data",
            "repeated_emotions": [],
            "recent_state": None,
        }

    ordered = _sorted(records)

    recent = ordered[
        -max(1, recent_n):
    ]

    frequency = {}

    for record in ordered:
        emotion = record.dominant_emotion

        frequency[emotion] = (
            frequency.get(emotion, 0) + 1
        )

    max_count = max(
        frequency.values()
    )

    dominant = sorted([
        emotion
        for emotion, count in frequency.items()
        if count == max_count
    ])

    avg_all = (
        sum(r.intensity for r in ordered)
        / len(ordered)
    )

    avg_recent = (
        sum(r.intensity for r in recent)
        / len(recent)
    )

    delta = avg_recent - avg_all

    positive_count = sum(
        r.polarity == "positive"
        for r in ordered
    )

    negative_count = sum(
        r.polarity == "negative"
        for r in ordered
    )

    if len(ordered) >= 2:

        midpoint = max(
            1,
            len(ordered) // 2
        )

        first = ordered[:midpoint]
        second = ordered[midpoint:]

        first_p = (
            sum(r.polarity_score for r in first)
            / len(first)
        )

        second_p = (
            sum(r.polarity_score for r in second)
            / len(second)
        )

        difference = second_p - first_p

        if difference > 0.05:
            polarity_trend = "improving"

        elif difference < -0.05:
            polarity_trend = "worsening"

        else:
            polarity_trend = "stable"

    else:
        polarity_trend = "insufficient_data"

    recent_emotions = [
        r.dominant_emotion
        for r in recent
    ]

    repeated = sorted({
        emotion
        for emotion in recent_emotions
        if recent_emotions.count(emotion) >= 2
    })

    return {
        "sample_count": len(ordered),
        "emotion_frequency": frequency,
        "dominant_emotions": dominant,
        "average_intensity": round(
            avg_all, 4
        ),
        "recent_average_intensity": round(
            avg_recent, 4
        ),
        "intensity_delta": round(
            delta, 4
        ),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "polarity_trend": polarity_trend,
        "repeated_emotions": repeated,
        "recent_state": ordered[-1].to_dict(),
    }


def history_influence(
    records: List[EmotionHistoryRecord],
    current_emotion: str
) -> float:

    if not records:
        return 0.0

    recent = _sorted(records)[-10:]

    matches = sum(
        r.dominant_emotion == current_emotion
        for r in recent
    )

    return matches / len(recent)


def build_history_record(
    user_id: str,
    emotional_state
) -> EmotionHistoryRecord:

    return EmotionHistoryRecord(
        user_id=user_id,
        timestamp=datetime.now(
            timezone.utc
        ).isoformat(),

        dominant_emotion=(
            emotional_state.dominant_emotion
        ),

        confidence=float(
            emotional_state.emotion_confidence
        ),

        intensity=float(
            emotional_state.intensity
        ),

        polarity=emotional_state.polarity,

        polarity_score=float(
            emotional_state.polarity_score
        ),

        triggered_emotions=list(
            emotional_state.triggered_emotions
        ),

        emotion_scores=dict(
            emotional_state.emotion_probabilities
        ),
    )