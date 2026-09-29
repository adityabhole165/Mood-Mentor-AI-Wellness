"""MoodMentor API (M3 Task 10 + M4 extended endpoints).

Run with:
    uvicorn src.api:app --reload
"""
from __future__ import annotations
import io
import json
import math
import os
import threading
import zipfile
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .recommendation_data import UserProfile, DEFAULT_WELLNESS_CONTENT
from .database import save_analysis, save_feedback, count_rows, delete_user_data
from .security import validate_user_id, validate_event

DB_PATH = os.getenv("MOOD_MENTOR_DB", os.path.join("data", "mood_mentor.db"))
HISTORY_PATH = os.getenv("MOOD_MENTOR_HISTORY", os.path.join("data", "emotion_history.csv"))
FEEDBACK_PATH = os.getenv("MOOD_MENTOR_FEEDBACK", os.path.join("data", "recommendation_feedback.csv"))
INTERACTIONS_PATH = os.getenv("MOOD_MENTOR_INTERACTIONS", os.path.join("data", "interactions.csv"))

app = FastAPI(title="MoodMentor API", version="3.0")

# Without this, every browser-based call from the React UI (a different
# origin than the API, e.g. http://localhost:5173 -> http://localhost:8000)
# is blocked by the browser before your code ever sees it. Tighten
# allow_origins to your real frontend URL(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv(
        "MOOD_MENTOR_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_warmup_state = {"status": "pending"}  # pending -> warming -> ready | failed


def _warm_models():
    """Load the default BERT model + semantic matcher once, in the background,
    so they're already cached (see model_cache.py) before the first /analyze
    request arrives. This is what actually fixes request timeouts -- without
    it, whichever request happens to be first pays the full model-load cost.
    """
    _warmup_state["status"] = "warming"
    try:
        from . import emotion_bert
        from .model_cache import get_emotion_model, get_semantic_matcher

        get_emotion_model("BERT", emotion_bert.DEFAULT_SAVE_DIR)
        get_semantic_matcher("all-MiniLM-L6-v2", DEFAULT_WELLNESS_CONTENT)
        _warmup_state["status"] = "ready"
    except Exception as exc:  # model files may not exist yet in a fresh checkout
        _warmup_state["status"] = f"failed: {exc}"


@app.on_event("startup")
def _on_startup():
    # Non-blocking: uvicorn can start accepting connections immediately;
    # early requests just see warmup_status != "ready" on /health.
    threading.Thread(target=_warm_models, daemon=True).start()


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    user_id: str = "demo_user"
    top_k: int = Field(default=5, ge=1, le=20)
    model_type: str = "BERT"
    preferred_types: list[str] = []
    preferred_tags: list[str] = []
    blocked_content_ids: list[str] = []
    persist_history: bool = True

class FeedbackRequest(BaseModel):
    user_id: str
    content_id: str
    event: Literal["view", "helpful", "complete", "accept", "reject", "skip", "hide", "rate"]
    rating: float | None = Field(default=None, ge=1, le=5)
    preference_changes: dict = {}

@app.get("/health")
def health():
    return {"status": "ok", "database": DB_PATH, "model_warmup": _warmup_state["status"]}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        validate_user_id(req.user_id)
        from .pipeline_v4 import run_milestone3_part2
        profile = UserProfile(req.user_id, req.preferred_types, req.preferred_tags, set(req.blocked_content_ids))
        result = run_milestone3_part2(
            req.text, profile, interactions_csv=INTERACTIONS_PATH,
            emotion_history_csv=HISTORY_PATH, feedback_csv=FEEDBACK_PATH,
            top_k=req.top_k, model_type=req.model_type, persist_history=req.persist_history,
        )
        if not result.get("is_valid"):
            raise HTTPException(status_code=400, detail=result.get("error", "Invalid input"))
        analysis_id = save_analysis(DB_PATH, req.user_id, result)
        result["analysis_id"] = analysis_id
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    from .recommendation_feedback import record_feedback
    try:
        validate_user_id(req.user_id)
        validate_event(req.event)
        record = record_feedback(
            FEEDBACK_PATH, req.user_id, req.content_id, req.event,
            rating=req.rating, preference_changes=req.preference_changes,
        )
        payload = record.to_dict()
        payload["timestamp"] = payload["timestamp"] or datetime.now(timezone.utc).isoformat()
        feedback_id = save_feedback(DB_PATH, payload)
        payload["database_id"] = feedback_id
        return payload
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/stats")
def stats():
    return {table: count_rows(DB_PATH, table) for table in ("analysis_results", "recommendations", "feedback", "emotion_history")}


# --- Milestone 4 "extended" endpoints -------------------------------------
# These wire up the already-existing helpers in m4_dashboard.py / database.py /
# reporting.py to the routes client.js has been calling since the start.

GRANULARITY_TO_FREQ = {"daily": "D", "weekly": "W", "monthly": "M"}


def _clean_value(v):
    """Make a pandas/py value JSON-safe: NaN -> None, Timestamp -> isoformat str."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "isoformat"):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return v


def _df_to_records(df) -> list[dict]:
    if df is None or df.empty:
        return []
    return [{k: _clean_value(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]


def _enrich_recommendations(df):
    """Pull `title` out of the stored result_json so the UI doesn't have to."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "result_json" in out.columns:
        def _title(x):
            try:
                return json.loads(x).get("title", "")
            except Exception:
                return ""
        out["title"] = out["result_json"].apply(_title)
    return out


@app.get("/history/emotions")
def history_emotions(user_id: str):
    from . import m4_dashboard
    try:
        validate_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    df = m4_dashboard.load_emotion_history_df(HISTORY_PATH, user_id)
    return {"user_id": user_id, "items": _df_to_records(df)}


@app.get("/history/recommendations")
def history_recommendations(user_id: str):
    from . import m4_dashboard
    try:
        validate_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    df = _enrich_recommendations(m4_dashboard.load_recommendation_history(DB_PATH, user_id))
    return {"user_id": user_id, "items": _df_to_records(df)}


@app.get("/history/feedback")
def history_feedback(user_id: str):
    from . import m4_dashboard
    try:
        validate_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    df = m4_dashboard.load_feedback_history(DB_PATH, user_id)
    return {"user_id": user_id, "items": _df_to_records(df)}


@app.get("/trends")
def trends(user_id: str, granularity: str = "daily"):
    from . import m4_dashboard
    try:
        validate_user_id(user_id)
        freq = GRANULARITY_TO_FREQ.get(granularity.lower())
        if freq is None:
            raise ValueError(f"Unsupported granularity: {granularity!r}. Use daily, weekly, or monthly.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history_df = m4_dashboard.load_emotion_history_df(HISTORY_PATH, user_id)
    trend_df = m4_dashboard.aggregate_emotion_trends(history_df, freq)
    freq_df = m4_dashboard.emotion_frequency(history_df)
    return {
        "user_id": user_id,
        "granularity": granularity.lower(),
        "trend": _df_to_records(trend_df),
        "emotion_frequency": _df_to_records(freq_df),
    }


@app.get("/reports")
def reports(user_id: str, format: str = "csv"):
    from . import m4_dashboard, reporting
    try:
        validate_user_id(user_id)
        fmt = format.lower()
        if fmt not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot = m4_dashboard.dashboard_snapshot(HISTORY_PATH, DB_PATH, user_id)

    if fmt == "json":
        return {
            "user_id": user_id,
            "history": _df_to_records(snapshot["history"]),
            "daily_trends": _df_to_records(snapshot["daily_trends"]),
            "emotion_frequency": _df_to_records(snapshot["emotion_frequency"]),
            "recommendations": _df_to_records(_enrich_recommendations(snapshot["recommendations"])),
            "feedback": _df_to_records(snapshot["feedback"]),
        }

    # format == "csv": bundle the 4 CSVs reporting.py already builds into one zip
    bundle = reporting.build_csv_bundle(snapshot)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in bundle.items():
            zf.writestr(name, content)
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{user_id}_report.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.delete("/users/{user_id}/data")
def delete_user_data_route(user_id: str):
    try:
        validate_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    deleted = delete_user_data(DB_PATH, user_id)
    return {"user_id": user_id, "deleted": deleted}