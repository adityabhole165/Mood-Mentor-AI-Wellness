"""Milestone 3 Task 2: ML-based personalization with consistent features."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

FEATURE_NAMES = [
    "emotion_relevance", "emotion_intensity", "user_preference", "content_similarity",
    "collaborative_score", "history_affinity", "novelty", "negative_severity",
]

def feature_vector(**kwargs):
    return [float(kwargs.get(name, 0.0)) for name in FEATURE_NAMES]

def feature_dict(values):
    return {name: float(values[i]) for i, name in enumerate(FEATURE_NAMES)}

@dataclass
class PersonalizedModel:
    model: object = None
    baseline: float = 0.0
    fitted: bool = False

    def fit(self, X, y):
        if len(X) < 8 or RandomForestRegressor is None:
            self.baseline = float(np.mean(y)) if len(y) else 0.0
            return self
        self.model = RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=2, random_state=42, n_jobs=-1)
        self.model.fit(np.asarray(X), np.asarray(y)); self.fitted = True
        return self

    def predict_one(self, features):
        value = self.model.predict([features])[0] if self.fitted and self.model is not None else self.baseline
        return float(np.clip(value, 0.0, 1.0))

def train_personalized_model(interactions) -> PersonalizedModel:
    X, y = [], []
    for x in interactions:
        if len(x.feature_vector) == len(FEATURE_NAMES):
            X.append([x.feature_vector[n] for n in FEATURE_NAMES]); y.append(float(np.clip(x.reward, 0, 1)))
    return PersonalizedModel().fit(X, y)
