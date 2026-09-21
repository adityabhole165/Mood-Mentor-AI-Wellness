"""CSV persistence for wellness content and recommendation interactions."""
from __future__ import annotations
import csv, json, os
from typing import Iterable, List
from .recommendation_data import WellnessContent, Interaction

LIST_SEP = "|"

def _split(value: str) -> List[str]:
    return [x.strip() for x in (value or "").split(LIST_SEP) if x.strip()]

def load_wellness_content(path: str) -> List[WellnessContent]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wellness content file not found: {path}")
    result = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            result.append(WellnessContent(
                content_id=row["content_id"].strip(),
                title=row["title"].strip(),
                description=row["description"].strip(),
                content_type=row["content_type"].strip(),
                tags=_split(row.get("tags", "")),
                emotions=_split(row.get("emotions", "")),
                url=row.get("url", "").strip(),
                duration_minutes=int(row.get("duration_minutes") or 5),
                intensity_level=int(row.get("intensity_level") or 1),
            ))
    return result

def load_interactions(path: str) -> List[Interaction]:
    if not os.path.exists(path):
        return []
    result = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_scores = row.get("emotion_scores", "") or "{}"
            try:
                emotion_scores = json.loads(raw_scores)
            except json.JSONDecodeError:
                emotion_scores = {}
            raw_features = row.get("feature_vector", "") or "{}"
            try:
                features = json.loads(raw_features)
            except json.JSONDecodeError:
                features = {}
            result.append(Interaction(
                user_id=row["user_id"].strip(),
                content_id=row["content_id"].strip(),
                reward=float(row.get("reward") or 0.0),
                interaction_type=row.get("interaction_type", "").strip(),
                timestamp=row.get("timestamp", "").strip(),
                emotion_scores={str(k): float(v) for k, v in emotion_scores.items()},
                emotion_intensity=float(row.get("emotion_intensity") or 0.0),
                feature_vector={str(k): float(v) for k, v in features.items()},
            ))
    return result

def append_interaction(path: str, interaction: Interaction) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["user_id", "content_id", "reward", "interaction_type", "timestamp", "emotion_scores", "emotion_intensity", "feature_vector"]
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "user_id": interaction.user_id,
            "content_id": interaction.content_id,
            "reward": interaction.reward,
            "interaction_type": interaction.interaction_type,
            "timestamp": interaction.timestamp,
            "emotion_scores": json.dumps(interaction.emotion_scores),
            "emotion_intensity": interaction.emotion_intensity,
            "feature_vector": json.dumps(interaction.feature_vector),
        })

def save_default_content(path: str, contents: Iterable[WellnessContent] = None) -> None:
    from .recommendation_data import DEFAULT_WELLNESS_CONTENT
    contents = list(contents or DEFAULT_WELLNESS_CONTENT)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["content_id", "title", "description", "content_type", "tags", "emotions", "url", "duration_minutes", "intensity_level"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for c in contents:
            writer.writerow({**c.to_dict(), "tags": LIST_SEP.join(c.tags), "emotions": LIST_SEP.join(c.emotions)})
