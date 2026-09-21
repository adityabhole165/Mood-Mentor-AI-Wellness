import json

from src.data_loader import (
    _split,
    append_interaction,
    load_interactions,
    load_wellness_content,
    save_default_content,
)
from src.recommendation_data import Interaction, DEFAULT_WELLNESS_CONTENT


def test_split_pipe_separated_values():
    assert _split(" calm | stress || ") == ["calm", "stress"]
    assert _split("") == []


def test_save_and_load_wellness_content(tmp_path):
    path = tmp_path / "wellness.csv"

    save_default_content(str(path))
    loaded = load_wellness_content(str(path))

    assert len(loaded) == len(DEFAULT_WELLNESS_CONTENT)
    assert loaded[0].content_id == DEFAULT_WELLNESS_CONTENT[0].content_id
    assert loaded[0].tags == DEFAULT_WELLNESS_CONTENT[0].tags


def test_append_and_load_interaction(tmp_path):
    path = tmp_path / "interactions.csv"

    interaction = Interaction(
        user_id="u1",
        content_id="W001",
        reward=1.0,
        interaction_type="helpful",
        emotion_scores={"fear": 0.8},
        emotion_intensity=0.7,
        feature_vector={"emotion_relevance": 0.8},
    )

    append_interaction(str(path), interaction)
    loaded = load_interactions(str(path))

    assert len(loaded) == 1
    assert loaded[0].user_id == "u1"
    assert loaded[0].content_id == "W001"
    assert loaded[0].reward == 1.0
    assert loaded[0].emotion_scores["fear"] == 0.8
    assert loaded[0].feature_vector["emotion_relevance"] == 0.8


def test_missing_interactions_file_returns_empty(tmp_path):
    assert load_interactions(str(tmp_path / "missing.csv")) == []
