from src.collaborative_filtering import CollaborativeFilter
from src.recommendation_data import Interaction


def interaction(user, content, reward):
    return Interaction(user, content, reward)


def test_cosine_identical_vectors_is_one():
    value = CollaborativeFilter._cosine({"W1": 1}, {"W1": 1})
    assert value == 1.0


def test_cosine_zero_vector_is_zero():
    assert CollaborativeFilter._cosine({}, {"W1": 1}) == 0.0


def test_score_uses_similar_users():
    interactions = [
        interaction("u1", "W1", 1.0),
        interaction("u2", "W1", 1.0),
        interaction("u2", "W2", 0.8),
        interaction("u3", "W3", 1.0),
    ]

    cf = CollaborativeFilter(interactions)
    score = cf.score("u1", "W2")

    assert score > 0
    assert score <= 1


def test_unknown_user_returns_zero():
    cf = CollaborativeFilter([])
    assert cf.score("unknown", "W1") == 0.0
