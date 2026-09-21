"""Milestone 3 Task 3: hybrid recommendation candidate generation."""
from __future__ import annotations
from datetime import datetime
import numpy as np
from .collaborative_filtering import CollaborativeFilter
from .personalized_recommender import PersonalizedModel, feature_vector

EMOTION_TO_TAGS = {
    "sadness": {"reflection", "positivity", "journaling", "calm"},
    "fear": {"breathing", "grounding", "anxiety", "relaxation"},
    "anger": {"anger-management", "pause", "reflection"},
    "joy": {"positivity", "motivation", "habits"},
    "surprise": {"grounding", "reflection", "self-awareness"},
    "disgust": {"grounding", "reflection", "calm"},
}

def _clamp(x): return max(0.0, min(1.0, float(x)))

class HybridRecommendationEngine:
    def __init__(self, contents, interactions=None, personalized_model=None):
        self.contents = contents
        self.interactions = interactions or []
        self.cf = CollaborativeFilter(self.interactions)
        self.personalized_model = personalized_model or PersonalizedModel()

    def _rule_score(self, state, content):
        if state.dominant_emotion in content.emotions: return 1.0
        wanted = set().union(*(EMOTION_TO_TAGS.get(e, set()) for e in state.triggered_emotions))
        return _clamp(len(wanted & set(content.tags)) / max(1, len(wanted)))

    def _preference_score(self, profile, content):
        type_match = 1.0 if content.content_type in profile.preferred_types else 0.0
        preferred = set(profile.preferred_tags)
        tag_match = len(set(content.tags) & preferred) / max(1, len(preferred))
        if not profile.preferred_types and not profile.preferred_tags: return 0.0
        return _clamp(0.6 * type_match + 0.4 * tag_match)

    def _emotion_similarity(self, state, content):
        return float(np.mean([state.emotion_probabilities.get(e, 0.0) for e in content.emotions])) if content.emotions else 0.0

    def _history_stats(self, user_id, content_id):
        seen = [x for x in self.interactions if x.user_id == user_id and x.content_id == content_id]
        if not seen: return 0.0, 1.0, 0
        avg = float(np.mean([x.reward for x in seen]))
        return avg, 0.0, len(seen)

    def generate_candidates(self, state, profile, semantic_scores=None):
        semantic_scores = semantic_scores or {}
        rows = []
        for c in self.contents:
            if c.content_id in profile.blocked_content_ids: continue
            emotion_rel = self._emotion_similarity(state, c)
            rule = self._rule_score(state, c)
            pref = self._preference_score(profile, c)
            semantic = _clamp(semantic_scores.get(c.content_id, 0.0))
            cf = self.cf.score(profile.user_id, c.content_id)
            history, novelty, exposure = self._history_stats(profile.user_id, c.content_id)
            features = feature_vector(emotion_relevance=emotion_rel, emotion_intensity=state.intensity,
                                      user_preference=pref, content_similarity=semantic,
                                      collaborative_score=cf, history_affinity=history, novelty=novelty,
                                      negative_severity=state.negative_emotion_load)
            ml = self.personalized_model.predict_one(features)
            rows.append({"content": c, "rule_score": rule, "content_similarity": semantic,
                         "preference_score": pref, "collaborative_score": cf, "emotion_relevance": emotion_rel,
                         "history_affinity": history, "novelty": novelty, "exposure_count": exposure,
                         "personalized_ml_score": ml, "features": features})
        return rows
