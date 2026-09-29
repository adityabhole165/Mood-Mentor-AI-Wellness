
import os
import sqlite3
import pandas as pd
from src.m4_dashboard import aggregate_emotion_trends, emotion_frequency, filter_records, search_dataframe
from src.database import connect, delete_user_data
from src.security import validate_user_id, validate_event, sanitize_search_query
from src.reporting import build_report_frame, build_csv_bundle


def sample_history():
    return pd.DataFrame([
        {"user_id":"u1","timestamp":"2026-01-01T10:00:00Z","dominant_emotion":"fear","intensity":0.8,"polarity_score":-0.5,"polarity":"negative","confidence":0.9},
        {"user_id":"u1","timestamp":"2026-01-02T10:00:00Z","dominant_emotion":"joy","intensity":0.4,"polarity_score":0.5,"polarity":"positive","confidence":0.8},
    ])


def test_daily_trends():
    df = sample_history()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = aggregate_emotion_trends(df, "D")
    assert len(out) == 2
    assert "avg_intensity" in out.columns


def test_frequency_filter_search():
    df = sample_history()
    assert emotion_frequency(df)["count"].sum() == 2
    assert len(filter_records(df, min_intensity=0.7)) == 1
    assert len(search_dataframe(df, "fear")) == 1


def test_security_validation():
    assert validate_user_id("demo_user-1") == "demo_user-1"
    assert validate_event("accept") == "accept"
    assert sanitize_search_query("  hello  ") == "hello"


def test_security_rejects_bad_values():
    import pytest
    with pytest.raises(ValueError):
        validate_user_id("../bad")
    with pytest.raises(ValueError):
        validate_event("drop_table")


def test_report_bundle():
    snap = {"history": sample_history(), "daily_trends": pd.DataFrame({"period":[]}), "recommendations": pd.DataFrame(), "feedback": pd.DataFrame()}
    assert "emotion_history.csv" in build_csv_bundle(snap)
    assert not build_report_frame(snap).empty


def test_user_data_deletion(tmp_path):
    db = str(tmp_path / "m.db")
    from src.database import save_feedback
    save_feedback(db, {"user_id":"u1","content_id":"W1","event":"accept","rating":5,"timestamp":"2026-01-01T00:00:00Z"})
    assert delete_user_data(db, "u1")["feedback"] == 1
