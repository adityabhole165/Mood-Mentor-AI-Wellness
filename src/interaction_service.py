"""Record user recommendation behavior for future personalization."""
from __future__ import annotations
from datetime import datetime, timezone
from .recommendation_data import Interaction
from .data_loader import append_interaction

REWARD_MAP = {"impression": 0.0, "view": 0.1, "click": 0.3, "start": 0.5, "complete": 0.8, "helpful": 1.0, "skip": 0.0, "hide": 0.0}

def reward_for_interaction(interaction_type: str, rating: float | None = None) -> float:
    if rating is not None:
        return max(0.0, min(1.0, float(rating) / 5.0))
    return REWARD_MAP.get(interaction_type.lower(), 0.0)

def record_interaction(path, user_id, content_id, interaction_type, emotion_scores=None, emotion_intensity=0.0, feature_vector=None, rating=None):
    interaction = Interaction(user_id=user_id, content_id=content_id, reward=reward_for_interaction(interaction_type, rating),
                              interaction_type=interaction_type, timestamp=datetime.now(timezone.utc).isoformat(),
                              emotion_scores=emotion_scores or {}, emotion_intensity=float(emotion_intensity),
                              feature_vector=feature_vector or {})
    append_interaction(path, interaction)
    return interaction
