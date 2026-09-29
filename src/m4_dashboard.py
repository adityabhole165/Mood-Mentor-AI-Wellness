
"""Milestone 4 dashboard/data helpers.

These functions are deliberately UI-light so they can be unit-tested without
starting Streamlit. They provide trend aggregation, filtering, history views,
and safe export data for the dashboard.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
import json
import pandas as pd


def _read_csv(path: str, columns=None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=columns or [])
    except Exception:
        return pd.DataFrame(columns=columns or [])
    return df


def load_emotion_history_df(path: str, user_id: str | None = None) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        return df
    if user_id and "user_id" in df.columns:
        df = df[df["user_id"].astype(str) == str(user_id)]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in ("intensity", "polarity_score", "confidence"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def aggregate_emotion_trends(df: pd.DataFrame, frequency: str = "D") -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["period", "avg_intensity", "avg_polarity", "samples"])
    work = df.copy()
    work["period"] = work["timestamp"].dt.floor(frequency)
    out = work.groupby("period", as_index=False).agg(
        avg_intensity=("intensity", "mean"),
        avg_polarity=("polarity_score", "mean"),
        samples=("period", "size"),
    )
    return out.sort_values("period").reset_index(drop=True)


def emotion_frequency(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dominant_emotion" not in df.columns:
        return pd.DataFrame(columns=["emotion", "count"])
    return (df["dominant_emotion"].astype(str).value_counts()
            .rename_axis("emotion").reset_index(name="count"))


def filter_records(
    df: pd.DataFrame,
    start=None,
    end=None,
    emotions: Iterable[str] | None = None,
    min_intensity: float | None = None,
    max_intensity: float | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "timestamp" in out:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    if start is not None and "timestamp" in out:
        out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC") if not pd.Timestamp(start).tzinfo else pd.Timestamp(start)]
    if end is not None and "timestamp" in out:
        out = out[out["timestamp"] <= pd.Timestamp(end, tz="UTC") if not pd.Timestamp(end).tzinfo else pd.Timestamp(end)]
    if emotions and "dominant_emotion" in out:
        out = out[out["dominant_emotion"].isin(list(emotions))]
    if min_intensity is not None and "intensity" in out:
        out = out[pd.to_numeric(out["intensity"], errors="coerce") >= float(min_intensity)]
    if max_intensity is not None and "intensity" in out:
        out = out[pd.to_numeric(out["intensity"], errors="coerce") <= float(max_intensity)]
    return out.reset_index(drop=True)


def load_recommendation_history(db_path: str, user_id: str | None = None) -> pd.DataFrame:
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        q = "SELECT * FROM recommendations"
        params = []
        if user_id:
            q += " WHERE user_id = ?"
            params.append(user_id)
        q += " ORDER BY created_at DESC"
        df = pd.read_sql_query(q, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_feedback_history(db_path: str, user_id: str | None = None) -> pd.DataFrame:
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        q = "SELECT * FROM feedback"
        params = []
        if user_id:
            q += " WHERE user_id = ?"
            params.append(user_id)
        q += " ORDER BY timestamp DESC"
        df = pd.read_sql_query(q, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def search_dataframe(df: pd.DataFrame, query: str, fields: list[str] | None = None) -> pd.DataFrame:
    if df.empty or not query:
        return df.copy()
    fields = fields or list(df.select_dtypes(include=["object"]).columns)
    mask = pd.Series(False, index=df.index)
    needle = str(query).lower()
    for field in fields:
        if field in df.columns:
            mask |= df[field].fillna("").astype(str).str.lower().str.contains(needle, regex=False)
    return df[mask].reset_index(drop=True)


def dashboard_snapshot(history_path: str, db_path: str, user_id: str) -> dict:
    history = load_emotion_history_df(history_path, user_id)
    recs = load_recommendation_history(db_path, user_id)
    feedback = load_feedback_history(db_path, user_id)
    trends = aggregate_emotion_trends(history, "D")
    return {
        "history": history,
        "daily_trends": trends,
        "emotion_frequency": emotion_frequency(history),
        "recommendations": recs,
        "feedback": feedback,
        "history_samples": len(history),
        "recommendation_samples": len(recs),
        "feedback_samples": len(feedback),
    }

def enrich_recommendation_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "result_json" in out.columns:
        def get_type(x):
            try:
                return json.loads(x).get("content_type", "")
            except Exception:
                return ""
        import json
        out["content_type"] = out["result_json"].apply(get_type)
    return out
