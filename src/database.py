"""Small SQLite persistence layer used by the Task 10 API boundary.

If a deployment already has a database, keep the table contract and replace
this adapter with the project's existing repository/ORM implementation.
"""
from __future__ import annotations
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    input_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dominant_emotion TEXT,
    intensity REAL,
    polarity TEXT,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    user_id TEXT NOT NULL,
    content_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL,
    FOREIGN KEY(analysis_id) REFERENCES analysis_results(id)
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    content_id TEXT NOT NULL,
    event TEXT NOT NULL,
    rating REAL,
    timestamp TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS emotion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    dominant_emotion TEXT NOT NULL,
    intensity REAL NOT NULL,
    polarity TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""

@contextmanager
def connect(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        yield conn
    finally:
        conn.close()


def save_analysis(path: str, user_id: str, result: dict) -> int:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    state = result.get("emotional_state", {})
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO analysis_results(user_id,input_text,created_at,dominant_emotion,intensity,polarity,result_json) VALUES(?,?,?,?,?,?,?)",
            (user_id, result.get("input_text", ""), now, state.get("dominant_emotion"), state.get("intensity", 0.0), state.get("polarity", "neutral"), json.dumps(result, default=str)),
        )
        analysis_id = cur.lastrowid
        for rec in result.get("recommendations", []):
            conn.execute(
                "INSERT INTO recommendations(analysis_id,user_id,content_id,rank,score,created_at,result_json) VALUES(?,?,?,?,?,?,?)",
                (analysis_id, user_id, rec["content_id"], rec["rank"], rec["score"], now, json.dumps(rec, default=str)),
            )
        for h in result.get("emotion_history", {}).get("new_records", []):
            conn.execute(
                "INSERT INTO emotion_history(user_id,timestamp,dominant_emotion,intensity,polarity,payload_json) VALUES(?,?,?,?,?,?)",
                (user_id, h.get("timestamp", now), h.get("dominant_emotion", ""), h.get("intensity", 0.0), h.get("polarity", "neutral"), json.dumps(h)),
            )
        conn.commit()
    return int(analysis_id)


def save_feedback(path: str, payload: dict) -> int:
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO feedback(user_id,content_id,event,rating,timestamp,payload_json) VALUES(?,?,?,?,?,?)",
            (payload["user_id"], payload["content_id"], payload["event"], payload.get("rating"), payload["timestamp"], json.dumps(payload, default=str)),
        )
        conn.commit()
        return int(cur.lastrowid)


def count_rows(path: str, table: str) -> int:
    if table not in {"analysis_results", "recommendations", "feedback", "emotion_history"}:
        raise ValueError("invalid table")
    with connect(path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def delete_user_data(path: str, user_id: str) -> dict:
    """Delete all persisted SQLite records for a user. Returns deleted counts."""
    with connect(path) as conn:
        result = {}
        for table in ("recommendations", "feedback", "emotion_history", "analysis_results"):
            cur = conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            result[table] = cur.rowcount
        conn.commit()
    return result
