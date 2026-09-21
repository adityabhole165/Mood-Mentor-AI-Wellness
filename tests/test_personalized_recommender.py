import numpy as np

from src.personalized_recommender import (
    FEATURE_NAMES,
    PersonalizedModel,
    feature_vector,
    feature_dict,
    train_personalized_model,
)
from src.recommendation_data import Interaction


def test_feature_vector_has_consistent_order():
    values = feature_vector(
        emotion_relevance=0.1,
        emotion_intensity=0.2,
        user_preference=0.3,
        content_similarity=0.4,
        collaborative_score=0.5,
        history_affinity=0.6,
        novelty=0.7,
        negative_severity=0.8,
    )

    assert len(values) == len(FEATURE_NAMES)
    assert values == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def test_feature_dict_round_trip():
    values = list(range(len(FEATURE_NAMES)))
    result = feature_dict(values)

    assert list(result.keys()) == FEATURE_NAMES
    assert list(result.values()) == [float(x) for x in values]


def test_small_dataset_uses_baseline():
    model = PersonalizedModel().fit([[0] * 8], [0.75])

    assert model.fitted is False
    assert model.baseline == 0.75
    assert model.predict_one([0] * 8) == 0.75


def test_empty_training_data_is_safe():
    model = PersonalizedModel().fit([], [])

    assert model.baseline == 0.0
    assert model.predict_one([0] * 8) == 0.0


def test_train_personalized_model_ignores_wrong_feature_lengths():
    valid_features = {name: 0.5 for name in FEATURE_NAMES}

    interactions = [
        Interaction("u1", "W1", 0.8, feature_vector=valid_features),
        Interaction("u2", "W2", 0.2, feature_vector={"bad": 1.0}),
    ]

    model = train_personalized_model(interactions)

    assert model.baseline == 0.8
