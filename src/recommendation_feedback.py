"""Milestone 3 Part 2 - Task 7: feedback capture and learning."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List

from .personalized_recommender import FEATURE_NAMES, PersonalizedModel


VALID_EVENTS = {
    "view", "accept", "reject", "skip", "complete", "helpful", "hide", "rate"
}


@dataclass
class FeedbackRecord:
    user_id: str
    content_id: str
    event: str
    timestamp: str
    rating: float | None = None
    preference_changes: Dict[str, object] | None = None
    feature_vector: Dict[str, float] | None = None
    recommendation_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


FIELDS = [
    "user_id", "content_id", "event", "timestamp", "rating",
    "preference_changes", "feature_vector", "recommendation_score"
]


def _reward(event: str, rating: float | None) -> float:
    if rating is not None:
        return max(0.0, min(1.0, float(rating) / 5.0))
    return {
        "view": 0.10, "accept": 0.80, "complete": 0.90,
        "helpful": 1.00, "reject": 0.00, "skip": 0.00, "hide": 0.00
    }.get(event.lower(), 0.0)


def record_feedback(
    path: str,
    user_id: str,
    content_id: str,
    event: str,
    *,
    rating: float | None = None,
    preference_changes: Dict[str, object] | None = None,
    feature_vector: Dict[str, float] | None = None,
    recommendation_score: float = 0.0,
) -> FeedbackRecord:
    event = event.lower().strip()
    if event not in VALID_EVENTS:
        raise ValueError(f"Unsupported feedback event: {event}")

    record = FeedbackRecord(
        user_id=user_id,
        content_id=content_id,
        event=event,
        timestamp=datetime.now(timezone.utc).isoformat(),
        rating=rating,
        preference_changes=preference_changes or {},
        feature_vector=feature_vector or {},
        recommendation_score=float(recommendation_score),
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            **record.to_dict(),
            "preference_changes": json.dumps(record.preference_changes or {}),
            "feature_vector": json.dumps(record.feature_vector or {}),
        })
    return record


def load_feedback(path: str, user_id: str | None = None) -> List[FeedbackRecord]:
    if not os.path.exists(path):
        return []
    result = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if user_id and row.get("user_id", "").strip() != user_id:
                continue
            try:
                changes = json.loads(row.get("preference_changes") or "{}")
                features = json.loads(row.get("feature_vector") or "{}")
            except json.JSONDecodeError:
                changes, features = {}, {}
            result.append(FeedbackRecord(
                user_id=row.get("user_id", "").strip(),
                content_id=row.get("content_id", "").strip(),
                event=row.get("event", "").strip(),
                timestamp=row.get("timestamp", "").strip(),
                rating=float(row["rating"]) if row.get("rating") else None,
                preference_changes=changes,
                feature_vector={str(k): float(v) for k, v in features.items()},
                recommendation_score=float(row.get("recommendation_score") or 0),
            ))
    return result


def feedback_summary(records: List[FeedbackRecord]) -> dict:
    counts: Dict[str, int] = {}
    ratings = []
    for r in records:
        counts[r.event] = counts.get(r.event, 0) + 1
        if r.rating is not None:
            ratings.append(r.rating)
    return {
        "total": len(records),
        "event_counts": counts,
        "average_rating": round(sum(ratings) / len(ratings), 3) if ratings else None,
        "acceptance_rate": round(
            counts.get("accept", 0) / max(1, counts.get("view", 0)), 4
        ),
    }


def train_model_from_feedback(records: List[FeedbackRecord]) -> PersonalizedModel:
    """Retrain the ranking model from explicit recommendation feedback."""
    X, y = [], []
    for r in records:
        if not r.feature_vector:
            continue
        if not all(name in r.feature_vector for name in FEATURE_NAMES):
            continue
        X.append([float(r.feature_vector[name]) for name in FEATURE_NAMES])
        y.append(_reward(r.event, r.rating))
    return PersonalizedModel().fit(X, y)
