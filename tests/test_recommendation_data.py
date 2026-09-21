from src.recommendation_data import (
    WellnessContent,
    UserProfile,
    Interaction,
    DEFAULT_WELLNESS_CONTENT,
)


def test_wellness_content_to_dict():
    content = WellnessContent(
        "X1", "Test", "Description", "breathing",
        ["calm"], ["fear"], "https://example.com", 5, 1
    )
    data = content.to_dict()

    assert data["content_id"] == "X1"
    assert data["tags"] == ["calm"]
    assert data["emotions"] == ["fear"]
    assert data["duration_minutes"] == 5


def test_user_profile_defaults_are_independent():
    a = UserProfile("u1")
    b = UserProfile("u2")

    a.preferred_tags.append("calm")
    a.blocked_content_ids.add("W001")

    assert b.preferred_tags == []
    assert b.blocked_content_ids == set()


def test_interaction_to_dict():
    interaction = Interaction(
        user_id="u1",
        content_id="W001",
        reward=0.8,
        interaction_type="complete",
        feature_vector={"novelty": 1.0},
    )
    data = interaction.to_dict()

    assert data["user_id"] == "u1"
    assert data["reward"] == 0.8
    assert data["feature_vector"]["novelty"] == 1.0


def test_default_catalog_is_populated():
    assert len(DEFAULT_WELLNESS_CONTENT) >= 7
    assert {c.content_id for c in DEFAULT_WELLNESS_CONTENT} == {
        "W001", "W002", "W003", "W004", "W005", "W006", "W007"
    }
