import csv

from src.interaction_service import (
    REWARD_MAP,
    record_interaction,
    reward_for_interaction,
)
from src.data_loader import load_interactions


def test_known_interaction_rewards():
    for name, expected in REWARD_MAP.items():
        assert reward_for_interaction(name) == expected


def test_unknown_interaction_defaults_to_zero():
    assert reward_for_interaction("something_unknown") == 0.0


def test_rating_is_converted_to_0_1():
    assert reward_for_interaction("helpful", rating=5) == 1.0
    assert reward_for_interaction("helpful", rating=2.5) == 0.5
    assert reward_for_interaction("helpful", rating=10) == 1.0
    assert reward_for_interaction("helpful", rating=-1) == 0.0


def test_record_interaction_persists_data(tmp_path):
    path = tmp_path / "interactions.csv"

    interaction = record_interaction(
        str(path),
        user_id="u1",
        content_id="W001",
        interaction_type="complete",
        emotion_scores={"fear": 0.8},
        emotion_intensity=0.7,
        feature_vector={"emotion_relevance": 0.8},
    )

    assert interaction.reward == 0.8
    assert interaction.user_id == "u1"
    assert interaction.content_id == "W001"
    assert interaction.timestamp

    loaded = load_interactions(str(path))
    assert len(loaded) == 1
    assert loaded[0].interaction_type == "complete"
    assert loaded[0].emotion_scores["fear"] == 0.8
