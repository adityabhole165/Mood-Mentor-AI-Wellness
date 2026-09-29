from pathlib import Path

from fastapi.testclient import TestClient

import src.api as api


def test_api_health_and_stats(tmp_path, monkeypatch):
    db = str(Path(tmp_path) / "mood.db")
    monkeypatch.setattr(api, "DB_PATH", db)
    client = TestClient(api.app)
    assert client.get("/health").status_code == 200
    body = client.get("/stats").json()
    assert body["analysis_results"] == 0
    assert body["feedback"] == 0


def test_feedback_api_persists_csv_and_database(tmp_path, monkeypatch):
    db = str(Path(tmp_path) / "mood.db")
    feedback_csv = str(Path(tmp_path) / "feedback.csv")
    monkeypatch.setattr(api, "DB_PATH", db)
    monkeypatch.setattr(api, "FEEDBACK_PATH", feedback_csv)
    client = TestClient(api.app)
    response = client.post("/feedback", json={
        "user_id": "u1",
        "content_id": "W001",
        "event": "accept",
        "rating": 5,
        "preference_changes": {"preferred_tags": ["calm"]},
    })
    assert response.status_code == 200
    assert response.json()["database_id"] == 1
    stats = client.get("/stats").json()
    assert stats["feedback"] == 1
    assert Path(feedback_csv).exists()


def test_analyze_api_persists_analysis_and_recommendations(tmp_path, monkeypatch):
    db = str(Path(tmp_path) / "mood.db")
    monkeypatch.setattr(api, "DB_PATH", db)
    monkeypatch.setattr(api, "INTERACTIONS_PATH", str(Path(tmp_path) / "interactions.csv"))
    monkeypatch.setattr(api, "HISTORY_PATH", str(Path(tmp_path) / "history.csv"))
    monkeypatch.setattr(api, "FEEDBACK_PATH", str(Path(tmp_path) / "feedback.csv"))

    def fake_pipeline(*args, **kwargs):
        return {
            "is_valid": True,
            "input_text": "hello",
            "emotional_state": {"dominant_emotion": "joy", "intensity": 0.5, "polarity": "positive"},
            "recommendations": [{"content_id": "W001", "rank": 1, "score": 0.8}],
            "emotion_history": {"new_records": []},
        }

    import sys, types
    fake_module = types.ModuleType("src.pipeline_v4")
    fake_module.run_milestone3_part2 = fake_pipeline
    monkeypatch.setitem(sys.modules, "src.pipeline_v4", fake_module)

    client = TestClient(api.app)
    response = client.post("/analyze", json={"text": "hello", "user_id": "u1"})
    assert response.status_code == 200
    assert response.json()["analysis_id"] == 1
    stats = client.get("/stats").json()
    assert stats["analysis_results"] == 1
    assert stats["recommendations"] == 1


def test_feedback_openapi_exposes_allowed_events():
    schema = api.app.openapi()
    event_schema = schema["components"]["schemas"]["FeedbackRequest"]["properties"]["event"]
    assert set(event_schema["enum"]) == {
        "view", "accept", "reject", "skip", "complete", "helpful", "hide", "rate"
    }
