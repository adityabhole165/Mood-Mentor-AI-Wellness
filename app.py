"""
app.py
Mood Mentor - Streamlit UI (Milestones 1, 2, 3 and 4)

One app that walks a piece of text through the whole pipeline:

    Enter Text -> Text Preprocessing -> VADER Sentiment            (Milestone 1)
    -> BERT / DistilBERT Emotion Classification -> Confidence      (Milestone 2)
    -> Emotional State Analysis -> Semantic Matching
    -> Hybrid Recommendation (rules + collaborative + ML) -> Ranking
    -> User Feedback -> Personalization                            (Milestone 3)
    -> Dashboard / History / Reports / Privacy                     (Milestone 4)

Nothing in src/ is modified. Milestone 3 is wired stage-by-stage (the same
order as src/pipeline_v3.py) instead of calling run_milestone3() directly, so
that the BERT / DistilBERT models and the sentence-transformer are loaded ONCE
and cached, rather than reloaded on every click, and so every intermediate
result can be shown in the UI.

--------------------------------------------------------------------------
DEPLOYMENT NOTES
--------------------------------------------------------------------------
Run locally:
    streamlit run app.py

Environment variables (all optional -- sensible defaults are used if unset):
    MOOD_MENTOR_DATA_DIR      Folder for CSV/JSON/SQLite data files (default: ./data)
    MOOD_MENTOR_REPORTS_DIR   Folder for generated PDF reports     (default: ./reports)
    MOOD_MENTOR_BERT_DIR      Fine-tuned BERT model folder         (default: src default)
    MOOD_MENTOR_DISTIL_DIR    Fine-tuned DistilBERT model folder   (default: src default)
    MOOD_MENTOR_SEMANTIC_MODEL  sentence-transformers model name   (default: all-MiniLM-L6-v2)
    MOOD_MENTOR_LOG_LEVEL     Python logging level                 (default: INFO)

Before deploying, make sure the data/reports directories exist (or are
writable so the app can create them) and that the BERT/DistilBERT model
folders are either baked into the image or downloaded on startup.
--------------------------------------------------------------------------
"""

import json
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import run_pipeline
from src.ingestion import read_input_data
from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment
from src import emotion_bert
from src import emotion_distilbert
from src.emotion_dataset import EMOTIONS
from src.confidence import build_confidence_report
from src.emotional_state import analyze_emotional_state
from src.recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile
from src.data_loader import load_wellness_content, load_interactions, save_default_content
from src.semantic_matching import SemanticMatcher
from src import semantic_matching as _semantic_matching_module
from src.hybrid_recommender import HybridRecommendationEngine
from src.ranking import RecommendationRanker
from src.personalized_recommender import (
    FEATURE_NAMES,
    feature_dict,
    feature_vector,
    train_personalized_model,
)
from src.interaction_service import REWARD_MAP, record_interaction
from src.emotion_history import build_history_record, load_emotion_history, append_emotion_history, analyze_emotion_trend
from src.recommendation_feedback import record_feedback, load_feedback, feedback_summary, train_model_from_feedback
from src.recommendation_explainability import add_explanations
from src.recommendation_evaluation import evaluate_rankings, recommendation_diversity, compare_baseline_and_advanced, acceptance_for_predictions
from src.evaluation_runner import load_evaluation_cases, run_controlled_evaluation as run_task9_evaluation
from src.regression_tests import run_regression_tests
from src.m4_dashboard import dashboard_snapshot, filter_records, search_dataframe, aggregate_emotion_trends, enrich_recommendation_history
from src.reporting import generate_pdf_report, build_csv_bundle
from src.security import validate_user_id, sanitize_search_query


# ===========================================================
# LOGGING
# ===========================================================

logging.basicConfig(
    level=os.environ.get("MOOD_MENTOR_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mood_mentor")


# ===========================================================
# CONFIGURATION (env-overridable so the same image can be
# deployed against different data/model locations)
# ===========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MOOD_MENTOR_DATA_DIR", os.path.join(BASE_DIR, "data"))
REPORTS_DIR = os.environ.get("MOOD_MENTOR_REPORTS_DIR", os.path.join(BASE_DIR, "reports"))

DEFAULT_INTERACTIONS_PATH = os.path.join(DATA_DIR, "interactions.csv")
DEFAULT_FEEDBACK_PATH = os.path.join(DATA_DIR, "recommendation_feedback.csv")
DEFAULT_HISTORY_PATH = os.path.join(DATA_DIR, "emotion_history.csv")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "mood_mentor.db")
DEFAULT_EVAL_CASES_PATH = os.path.join(DATA_DIR, "m3_evaluation_cases.json")

DEFAULT_SEMANTIC_MODEL = os.environ.get("MOOD_MENTOR_SEMANTIC_MODEL", "all-MiniLM-L6-v2")
DEFAULT_BERT_DIR = os.environ.get("MOOD_MENTOR_BERT_DIR", emotion_bert.DEFAULT_SAVE_DIR)
DEFAULT_DISTIL_DIR = os.environ.get("MOOD_MENTOR_DISTIL_DIR", emotion_distilbert.DEFAULT_SAVE_DIR)

os.makedirs(DATA_DIR, exist_ok=True)

MODE_CHAT = "💬 Chat Text"
MODE_TXT = "📄 TXT File"
MODE_CSV = "📊 CSV File"

EXAMPLES = {
    "😰 Excited but nervous": "I am excited about the new opportunity but nervous about the outcome.",
    "😔 Low & overwhelmed": "I have been feeling really down and overwhelmed at work lately, nothing seems to go right.",
    "😠 Frustrated": "I'm furious that my manager took credit for my work again. I can't stop thinking about it.",
}


def init_session_state():
    """Ensures every key the app reads later exists, even on a fresh session."""
    defaults = {"analysis": None, "m3": None, "feedback_log": [], "chat_text": "", "m3_p2_eval": None}
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)


# ===========================================================
# SMALL UI HELPERS
# ===========================================================

def show_df(df, **kwargs):
    """st.dataframe at full width, across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, **kwargs)


def _set_chat_text(value: str):
    st.session_state["chat_text"] = value


# ===========================================================
# MODEL / RESOURCE LOADING (cached -- runs once per session/path)
# ===========================================================

@st.cache_resource(show_spinner=False)
def load_emotion_models(bert_dir: str, distil_dir: str):
    """Loads both fine-tuned models once. Returns (bert_model, bert_tok,
    distil_model, distil_tok, errors). A model that fails to load stays None
    and the reason is recorded in `errors`, so one missing model folder
    doesn't crash the whole app."""
    bert_model = bert_tok = None
    distil_model = distil_tok = None
    errors = {}

    try:
        bert_model, bert_tok = emotion_bert.load_trained_model(bert_dir)
    except Exception as e:
        logger.warning("Could not load BERT model from %s: %s", bert_dir, e)
        errors["BERT"] = str(e)

    try:
        distil_model, distil_tok = emotion_distilbert.load_trained_model(distil_dir)
    except Exception as e:
        logger.warning("Could not load DistilBERT model from %s: %s", distil_dir, e)
        errors["DistilBERT"] = str(e)

    return bert_model, bert_tok, distil_model, distil_tok, errors


@st.cache_resource(show_spinner=False)
def load_semantic_matcher(model_name: str):
    """Loads the sentence-transformer once. If it can't be loaded (no
    sentence-transformers installed, or no internet for the first download),
    falls back to the TF-IDF matcher that src/semantic_matching.py already
    supports, and reports which backend is active."""
    try:
        matcher = SemanticMatcher(model_name)
        if matcher.model is not None:
            return matcher, f"sentence-transformers ({model_name})", None
        return matcher, "TF-IDF fallback (sentence-transformers not installed)", None
    except Exception as e:
        # Model download / load failed: temporarily hide SentenceTransformer so
        # SemanticMatcher takes its built-in TF-IDF path (no src/ edits needed).
        logger.warning("Semantic model '%s' failed to load, falling back to TF-IDF: %s", model_name, e)
        original = _semantic_matching_module.SentenceTransformer
        _semantic_matching_module.SentenceTransformer = None
        try:
            matcher = SemanticMatcher(model_name)
        finally:
            _semantic_matching_module.SentenceTransformer = original
        return matcher, "TF-IDF fallback (sentence-transformer could not be loaded)", str(e)


@st.cache_data(show_spinner=False)
def parse_content_csv(data: bytes):
    """Parse an uploaded wellness-content CSV with src.data_loader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return load_wellness_content(path)
    finally:
        os.unlink(path)


@st.cache_data(show_spinner=False)
def content_template_csv() -> bytes:
    """The built-in library as a CSV, so users can copy the expected format."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        path = tmp.name
    try:
        save_default_content(path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


# ===========================================================
# MILESTONE 1 + 2 -- ANALYSIS
# ===========================================================

def ingest_and_analyze(mode: str, text: str = None, uploaded=None):
    """Milestone 1: ingestion -> preprocessing -> VADER, via src/pipeline.py."""
    if mode == MODE_CHAT:
        return run_pipeline("raw_text", text)

    suffix, source_type = (".txt", "txt_file") if mode == MODE_TXT else (".csv", "csv_file")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        return run_pipeline(source_type, tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_emotion_stage(rows, bert_model, bert_tok, distil_model, distil_tok):
    """
    rows: list of (id, text) for VALID records only.
    Milestone 2: BERT + DistilBERT emotion classification with confidence
    scores; uses compare_predictions() when both models are loaded, and falls
    back to whichever single model is available.
    """
    results = []
    for rid, text in rows:
        entry = {"id": rid, "text": text}
        try:
            if bert_model and distil_model:
                cmp = emotion_distilbert.compare_predictions(
                    text, bert_model, bert_tok, distil_model, distil_tok
                )
                entry["bert"] = cmp["bert"]
                entry["distilbert"] = cmp["distilbert"]
                entry["agree_on_primary"] = cmp["agree_on_primary"]
            elif bert_model:
                entry["bert"] = emotion_bert.predict(text, bert_model, bert_tok).to_dict()
            elif distil_model:
                entry["distilbert"] = emotion_distilbert.predict(text, distil_model, distil_tok).to_dict()
            else:
                entry["error"] = "No emotion model is loaded."
        except Exception as e:
            # Same rule as pipeline.py: surface the error on this row, don't take down the batch.
            logger.exception("Emotion stage failed for record %s", rid)
            entry["error"] = f"Emotion stage error: {e}"
        results.append(entry)
    return results


def build_combined_report(m1_df: pd.DataFrame, emotion_results: list) -> pd.DataFrame:
    """Merges the Milestone 1 sentiment rows with the Milestone 2 emotion results."""
    by_id = {r["id"]: r for r in emotion_results}
    rows = []
    for _, row in m1_df.iterrows():
        rec = row.to_dict()
        er = by_id.get(row["id"])
        if er and not er.get("error"):
            if "bert" in er:
                b = er["bert"]
                rec["bert_primary_emotion"] = b["primary_emotion"]
                rec["bert_confidence"] = b["primary_confidence"]
                for e in EMOTIONS:
                    rec[f"bert_{e}"] = b["scores"][e]
            if "distilbert" in er:
                d = er["distilbert"]
                rec["distilbert_primary_emotion"] = d["primary_emotion"]
                rec["distilbert_confidence"] = d["primary_confidence"]
                for e in EMOTIONS:
                    rec[f"distilbert_{e}"] = d["scores"][e]
            if "agree_on_primary" in er:
                rec["models_agree"] = er["agree_on_primary"]
        rows.append(rec)
    return pd.DataFrame(rows)


# ===========================================================
# MILESTONE 3 -- RECOMMENDATION STAGE
# ===========================================================

def m3_signature(text: str, cfg: dict) -> str:
    """Identifies the inputs + settings a recommendation result was built from,
    so the UI can tell when it has gone stale."""
    return json.dumps([
        text, cfg["m3_model_name"], cfg["semantic_model_name"], cfg["top_k"], cfg["threshold"],
        cfg["user_id"], sorted(cfg["preferred_types"]), sorted(cfg["preferred_tags"]),
        sorted(cfg["blocked"]), cfg["content_key"], cfg["use_history"], cfg["interactions_path"], cfg["feedback_path"], cfg["history_path"],
    ])


def generate_m3(text: str, cfg: dict, models: tuple, persist_history: bool = True):
    """
    Milestone 3, stage by stage (same order as src/pipeline_v3.run_milestone3):
      preprocess -> VADER -> BERT/DistilBERT -> confidence
      -> Task 1 emotional state -> Task 5 semantic matching
      -> Task 2 personalized ML + collaborative filtering -> Task 3 hybrid candidates
      -> Task 4 ranking
    Stores the outcome in st.session_state["m3"].
    """
    bert_model, bert_tok, distil_model, distil_tok = models

    record = read_input_data("raw_text", text)[0]
    if not record.is_valid:
        st.session_state["m3"] = {"is_valid": False, "error": record.error, "signature": m3_signature(text, cfg)}
        return

    # --- Milestone 1 + 2 building blocks -------------------------------------
    processed = preprocess_text(record.raw_text)
    sentiment = analyze_sentiment(processed.cleaned_text)

    if cfg["m3_model_name"] == "BERT":
        model, tokenizer, predict_fn = bert_model, bert_tok, emotion_bert.predict
    else:
        model, tokenizer, predict_fn = distil_model, distil_tok, emotion_distilbert.predict
    if model is None:
        raise RuntimeError(
            f"The {cfg['m3_model_name']} model isn't loaded. Train it with "
            f"`python -m src.train_models` or fix its folder in the sidebar, "
            f"or pick the other model for Milestone 3."
        )

    prediction = predict_fn(processed.cleaned_text, model, tokenizer)
    confidence = build_confidence_report(prediction.scores)

    # --- Milestone 3 ---------------------------------------------------------
    state = analyze_emotional_state(confidence.all_scores, sentiment.compound)

    # Milestone 3 Part 2 — emotional history and trend.
    history_records = load_emotion_history(cfg["history_path"], cfg["user_id"])
    trend = analyze_emotion_trend(history_records)
    if persist_history:
        try:
            append_emotion_history(cfg["history_path"], build_history_record(cfg["user_id"], state))
            history_records = load_emotion_history(cfg["history_path"], cfg["user_id"])
            trend = analyze_emotion_trend(history_records)
        except Exception as history_error:
            logger.warning("Could not persist emotion history: %s", history_error)
            st.warning(f"Emotion history could not be saved: {history_error}")

    contents = cfg["contents"]
    matcher, backend, backend_error = load_semantic_matcher(cfg["semantic_model_name"])
    matcher.fit(contents)  # re-encode so a changed content library is never scored against stale vectors
    semantic_scores = matcher.match_emotional_state(state, record.raw_text, contents)

    interactions = load_interactions(cfg["interactions_path"]) if cfg["use_history"] else []
    personalized_model = train_personalized_model(interactions)
    feedback_records = load_feedback(cfg["feedback_path"], cfg["user_id"])
    feedback_model = train_model_from_feedback(feedback_records)
    feedback_stats = feedback_summary(feedback_records)
    usable_for_ml = sum(1 for x in interactions if len(x.feature_vector) == len(FEATURE_NAMES))

    stored_tags = set(cfg["preferred_tags"])
    stored_types = set(cfg["preferred_types"])
    for fb in feedback_records:
        changes = fb.preference_changes or {}
        stored_tags.update(str(x) for x in changes.get("preferred_tags", []) if x)
        stored_types.update(str(x) for x in changes.get("preferred_types", []) if x)
    profile = UserProfile(
        user_id=cfg["user_id"],
        preferred_types=sorted(stored_types),
        preferred_tags=sorted(stored_tags),
        blocked_content_ids=set(cfg["blocked"]),
    )

    engine = HybridRecommendationEngine(contents, interactions, personalized_model, feedback_model)
    candidates = engine.generate_candidates(state, profile, semantic_scores)

    ranker = RecommendationRanker(low_relevance_threshold=cfg["threshold"])
    ranked = ranker.rank(candidates, state, top_k=cfg["top_k"])
    ranked_dicts = add_explanations(
        [r.to_dict() for r in ranked],
        state.to_dict(),
        trend,
    )

    candidate_rows = []
    for row in candidates:
        candidate_rows.append({
            "content_id": row["content"].content_id,
            "title": row["content"].title,
            "final_score": round(ranker.score_row(row, state), 4),
            "personalized_ml": round(row["personalized_ml_score"], 4),
            "feedback_ml": round(row.get("feedback_ml_score", 0.0), 4),
            "feedback_learning_active": bool(row.get("feedback_learning_active", False)),
            "emotion_relevance": round(row["emotion_relevance"], 4),
            "semantic_similarity": round(row["content_similarity"], 4),
            "user_preference": round(row["preference_score"], 4),
            "collaborative": round(row["collaborative_score"], 4),
            "history_affinity": round(row["history_affinity"], 4),
            "rule_score": round(row["rule_score"], 4),
            "novelty": round(row["novelty"], 4),
        })
    candidate_rows.sort(key=lambda r: -r["final_score"])

    st.session_state["m3"] = {
        "is_valid": True,
        "signature": m3_signature(text, cfg),
        "input_text": text,
        "cleaned_text": processed.cleaned_text,
        "processed_text": processed.processed_text,
        "sentiment": sentiment.to_dict(),
        "emotion_model": cfg["m3_model_name"],
        "emotion": prediction.to_dict(),
        "confidence": confidence.to_dict(),
        "emotional_state": state.to_dict(),
        "emotion_history": {"records": len(history_records), "trend": trend},
        "semantic_backend": backend,
        "semantic_backend_error": backend_error,
        "semantic_similarity": semantic_scores,
        "recommendations": ranked_dicts,
        "candidates": candidate_rows,
        "features": {row["content"].content_id: feature_dict(row["features"]) for row in candidates},
        "ranking_weights": ranked[0].components["ranking_weights"] if ranked else ranker._weights(state.intensity),
        "personalization": {
            "n_interactions": len(interactions),
            "n_feedback": len(feedback_records),
            "usable_for_ml": usable_for_ml,
            "ml_fitted": bool(personalized_model.fitted),
            "baseline": personalized_model.baseline,
            "feedback_model_fitted": bool(feedback_model.fitted),
            "feedback_summary": feedback_stats,
        },
        "user_id": cfg["user_id"],
        "interactions_path": cfg["interactions_path"],
        "feedback_path": cfg["feedback_path"],
        "history_path": cfg["history_path"],
        "threshold": cfg["threshold"],
        "_contents_by_id": {c.content_id: c for c in contents},
    }


def log_feedback(content_id: str, interaction_type: str, rating=None, preference_changes=None):
    """Record legacy interaction plus Milestone 3 Part 2 feedback."""
    r = st.session_state.get("m3")
    if not r or not r.get("is_valid"):
        return
    try:
        interaction = record_interaction(
            r["interactions_path"], r["user_id"], content_id, interaction_type,
            emotion_scores=r["emotional_state"]["emotion_probabilities"],
            emotion_intensity=r["emotional_state"]["intensity"],
            feature_vector=r["features"].get(content_id, {}),
            rating=rating,
        )
        score = next(
            (float(x["score"]) for x in r["recommendations"]
             if x["content_id"] == content_id), 0.0
        )
        feedback = record_feedback(
            r["feedback_path"], r["user_id"], content_id, interaction_type,
            feature_vector=r["features"].get(content_id, {}),
            recommendation_score=score,
            rating=rating,
            preference_changes=preference_changes,
        )
        st.session_state["feedback_log"].append({
            "user_id": feedback.user_id,
            "content_id": feedback.content_id,
            "interaction": feedback.event,
            "reward": interaction.reward,
            "timestamp": feedback.timestamp,
        })
        st.toast(f"Logged “{interaction_type}” for {content_id}.")
    except Exception as e:
        logger.exception("Could not save feedback for %s / %s", content_id, interaction_type)
        st.toast(f"Could not save feedback: {e}")


def seed_demo_interactions(path: str, contents, n_users: int = 6, seed: int = 7) -> int:
    """
    Writes SYNTHETIC interaction history (clearly labelled demo_user_N) so that
    the collaborative-filtering and Random-Forest personalization stages have
    something to learn from during a demo. Appends only; never overwrites.
    """
    rng = np.random.default_rng(seed)
    types = sorted({c.content_type for c in contents})
    written = 0
    for u in range(1, n_users + 1):
        favourite_types = set(rng.choice(types, size=min(2, len(types)), replace=False))
        for c in contents:
            dominant = str(rng.choice(EMOTIONS))
            probs = {e: float(rng.uniform(0.0, 0.3)) for e in EMOTIONS}
            probs[dominant] = float(rng.uniform(0.6, 0.95))
            intensity = float(rng.uniform(0.3, 0.9))
            relevance = float(np.mean([probs[e] for e in c.emotions])) if c.emotions else 0.0
            preference = 1.0 if c.content_type in favourite_types else 0.0
            reward = float(np.clip(0.15 + 0.45 * relevance + 0.35 * preference + rng.normal(0, 0.05), 0, 1))
            features = feature_dict(feature_vector(
                emotion_relevance=relevance, emotion_intensity=intensity, user_preference=preference,
                content_similarity=float(rng.uniform(0.2, 0.8)), collaborative_score=0.0,
                history_affinity=0.0, novelty=1.0,
                negative_severity=float(np.mean([probs[e] for e in ("sadness", "anger", "fear", "disgust")])),
            ))
            kind = "helpful" if reward >= 0.7 else "complete" if reward >= 0.5 else "click" if reward >= 0.3 else "skip"
            record_interaction(path, f"demo_user_{u}", c.content_id, kind, emotion_scores=probs,
                               emotion_intensity=intensity, feature_vector=features, rating=reward * 5)
            written += 1
    return written


# ===========================================================
# RENDERING -- MILESTONE 1
# ===========================================================

def show_summary_metrics(summary, df):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Records", summary.get("total_samples", len(df)))
    with col2:
        st.metric("Average Compound", round(summary.get("avg_compound_score", 0) or 0, 3))
    with col3:
        st.metric("Positive", summary.get("positive_count", 0))
    with col4:
        st.metric("Neutral", summary.get("neutral_count", 0))
    with col5:
        st.metric("Negative", summary.get("negative_count", 0))


def render_milestone1(a):
    st.subheader("📈 Sentiment Summary (VADER)")
    show_summary_metrics(a["summary"], a["df"])

    invalid = int(a["summary"].get("invalid_samples", 0) or 0)
    if invalid:
        st.warning(f"{invalid} record(s) were invalid/skipped — see the `error` column below.")

    st.subheader("📋 Ingestion → Preprocessing → VADER results")
    show_df(a["df"])


# ===========================================================
# RENDERING -- MILESTONE 2
# ===========================================================

def _model_block(label: str, pred: dict):
    st.markdown(f"**{label}**")
    st.write(f"Primary emotion: **{pred['primary_emotion']}**  ·  confidence **{pred['primary_confidence']:.3f}**")
    triggered = ", ".join(pred["triggered_emotions"]) if pred["triggered_emotions"] else "—"
    st.caption(f"Triggered emotions (≥ {pred['threshold']:.2f} threshold): {triggered}")
    st.bar_chart(pd.Series(pred["scores"], name="confidence"))


def render_milestone2(a):
    st.subheader("🎭 Emotion Classification (BERT vs DistilBERT)")

    if not a["ran_emotion"]:
        st.info("Emotion analysis was switched off for this run. Tick "
                "**Run BERT + DistilBERT emotion analysis** in the sidebar and analyze again.")
        return

    for model_name, msg in a["load_errors"].items():
        st.warning(
            f"**{model_name}** could not be loaded ({msg}). Train it first with "
            f"`python -m src.train_models`, or fix the model folder path in the sidebar."
        )

    results = a["emotion_results"]
    if not results:
        st.info("No valid rows to run emotion analysis on.")
        return

    if a["emotion_capped"]:
        st.info(f"Emotion analysis limited to the first {len(results)} of {len(a['valid_rows'])} valid rows. "
                f"Raise the cap in the sidebar for more.")

    both = [r for r in results if "bert" in r and "distilbert" in r]
    if both:
        agree = sum(1 for r in both if r["agree_on_primary"])
        st.metric("BERT ↔ DistilBERT primary-emotion agreement", f"{agree}/{len(both)}")

    for r in results:
        preview = r["text"] if len(r["text"]) <= 90 else r["text"][:90] + "…"
        with st.expander(f"📝 {preview}", expanded=len(results) == 1):
            if r.get("error"):
                st.error(r["error"])
                continue

            has_bert, has_distil = "bert" in r, "distilbert" in r
            cols = st.columns(2) if (has_bert and has_distil) else st.columns(1)

            if has_bert:
                with cols[0]:
                    _model_block("BERT", r["bert"])
            if has_distil:
                with cols[1 if has_bert else 0]:
                    _model_block("DistilBERT", r["distilbert"])

            if "agree_on_primary" in r:
                if r["agree_on_primary"]:
                    st.success("✅ Both models agree on the primary emotion.")
                else:
                    st.warning("⚠️ Models disagree on the primary emotion.")

    if results:
        combined = build_combined_report(a["df"], results)
        st.download_button(
            "⬇️ Download Milestone 1 + 2 report (CSV)",
            combined.to_csv(index=False),
            file_name="mood_mentor_report.csv",
            mime="text/csv",
            key="dl_m12_report",
        )


# ===========================================================
# RENDERING -- MILESTONE 3
# ===========================================================

def render_emotional_state(r):
    s = r["emotional_state"]
    st.markdown("#### 1 · Deep emotional state")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dominant emotion", s["dominant_emotion"])
    c2.metric("Confidence", f"{s['emotion_confidence']:.2f}")
    c3.metric("Intensity", f"{s['intensity']:.2f}")
    c4.metric("Severity", s["severity"])
    c5.metric("Polarity", f"{s['polarity']} ({s['polarity_score']:+.2f})")

    left, right = st.columns([3, 2])
    with left:
        st.bar_chart(pd.Series(s["emotion_probabilities"], name="probability"))
    with right:
        triggered = ", ".join(s["triggered_emotions"]) if s["triggered_emotions"] else "—"
        st.write(f"**Triggered emotions:** {triggered}")
        st.write(f"**Mixed state:** {'yes' if s['mixed_state'] else 'no'}")
        st.write(f"**Positive load:** {s['positive_emotion_load']:.2f}  ·  "
                 f"**Negative load:** {s['negative_emotion_load']:.2f}")
        st.write(f"**Uncertainty:** {s['uncertainty']:.2f}")
        st.caption(f"Emotion source: {r['emotion_model']} · VADER compound {r['sentiment']['compound']:+.3f}")

    trend = r.get("emotion_history", {}).get("trend", {})
    with st.expander("📈 Emotional history & trend (Milestone 3 Part 2)"):
        if not trend or trend.get("sample_count", 0) == 0:
            st.info("No emotional history has been recorded yet.")
        else:
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("History samples", trend.get("sample_count", 0))
            h2.metric("Avg intensity", f"{trend.get('average_intensity', 0):.2f}")
            h3.metric("Recent intensity", f"{trend.get('recent_average_intensity', 0):.2f}")
            h4.metric("Polarity trend", trend.get("polarity_trend", "no_data"))
            repeated = ", ".join(trend.get("repeated_emotions", [])) or "—"
            st.write(f"**Repeated recent emotions:** {repeated}")

    if s["severity"] in ("high", "critical"):
        st.info(
            "This text reads as emotionally intense. The suggestions below are general wellness "
            "activities, not a substitute for professional support — if things feel like too much, "
            "consider reaching out to someone you trust or a mental-health professional."
        )


def render_recommendations(r):
    st.markdown(f"#### 2 · Top recommendations for `{r['user_id']}`")

    recs = r["recommendations"]
    if not recs:
        st.info(f"No wellness content cleared the relevance threshold ({r['threshold']:.2f}). "
                f"Try lowering the threshold in the sidebar or broadening the profile.")
        return

    contents = r["_contents_by_id"]
    view_key = f"viewed:{r['signature']}"
    if not st.session_state.get(view_key):
        for rec in recs:
            log_feedback(rec["content_id"], "view")
        st.session_state[view_key] = True
    for rec in recs:
        content = contents.get(rec["content_id"])
        with st.container(border=True):
            head, score_col = st.columns([4, 1])
            with head:
                st.markdown(f"### #{rec['rank']} · {rec['title']}")
                if content:
                    st.caption(
                        f"{content.content_type} · {content.duration_minutes} min · "
                        f"intensity {content.intensity_level} · tags: {', '.join(content.tags)}"
                    )
                    st.write(content.description)
                    if content.url:
                        st.markdown(f"[Open resource]({content.url})")
                explanation = rec.get("explanation", {})
                st.write("**Why:** " + " · ".join(explanation.get("reasons", rec["reasons"])))
                if explanation:
                    with st.expander("🔎 Explanation evidence"):
                        show_df(pd.DataFrame([explanation.get("evidence", {})]))
            with score_col:
                st.metric("Hybrid score", f"{rec['score']:.3f}")

            with st.expander("Score breakdown"):
                comps = {k: v for k, v in rec["components"].items()
                         if k not in ("ranking_weights", "emotional_intensity")}
                st.bar_chart(pd.Series(comps, name="component score"))

            st.caption("Feedback is stored and used for future personalization. After enough labelled feedback, the feedback model is blended into ranking.")
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            actions = (("👍 Helpful", "helpful"), ("✅ Complete", "complete"), ("✓ Accept", "accept"),
                       ("✕ Reject", "reject"), ("⏭ Skip", "skip"), ("🙈 Hide", "hide"))
            for col, (label, kind) in zip((b1, b2, b3, b4, b5, b6), actions):
                with col:
                    st.button(label, key=f"fb_{rec['rank']}_{rec['content_id']}_{kind}",
                              on_click=log_feedback, args=(rec["content_id"], kind))
            with st.expander("⭐ Rate / change preferences"):
                rating = st.slider("Rating", 1, 5, 3, key=f"rating_{rec['rank']}_{rec['content_id']}")
                changed_tags = st.multiselect("Preferred tags", sorted({t for c in contents.values() for t in c.tags}) if isinstance(contents, dict) else [], key=f"pref_{rec['rank']}_{rec['content_id']}")
                if st.button("Save rating & preferences", key=f"save_rating_{rec['rank']}_{rec['content_id']}"):
                    log_feedback(rec["content_id"], "rate", rating=float(rating),
                                 preference_changes={"preferred_tags": changed_tags})


def render_ranking_details(r):
    st.markdown("#### 3 · How the ranking was built")

    p = r["personalization"]
    if p.get("feedback_model_fitted"):
        st.success("Feedback learning: fitted and blended into the personalized ML score (35% feedback / 65% interaction model).")
    else:
        st.info("Feedback learning: waiting for at least 8 valid labelled feedback records before blending into ranking.")
    if p["ml_fitted"]:
        st.success(f"Personalized ML: Random Forest trained on {p['usable_for_ml']} logged interactions "
                   f"({p['n_interactions']} loaded).")
    else:
        st.info(
            f"Personalized ML: not enough history yet ({p['usable_for_ml']} usable interactions; needs 8+), "
            f"so a flat baseline of {p['baseline']:.2f} is used. Collaborative filtering and history affinity "
            f"also need logged interactions. Use the sidebar's demo-history button or the feedback buttons above."
        )

    st.caption(f"Semantic matching backend: **{r['semantic_backend']}**")
    if r.get("semantic_backend_error"):
        st.caption(f"Reason for fallback: {r['semantic_backend_error']}")

    col_w, col_c = st.columns([1, 3])
    with col_w:
        st.write("**Ranking weights** (scale with intensity)")
        show_df(pd.DataFrame({"weight": r["ranking_weights"]}).round(3))
    with col_c:
        st.write("**All candidates, scored** (before top-k and the relevance threshold)")
        show_df(pd.DataFrame(r["candidates"]))

    with st.expander("Pipeline trace — what each stage produced"):
        st.markdown(f"""
1. **Ingestion** → validated record
2. **Preprocessing** → cleaned text: `{r['cleaned_text']}`
3. **VADER** → {r['sentiment']['label']} (compound {r['sentiment']['compound']:+.3f})
4. **{r['emotion_model']} emotion model** → primary **{r['emotion']['primary_emotion']}** ({r['emotion']['primary_confidence']:.3f})
5. **Confidence report** → triggered at ≥ {r['confidence']['threshold']:.2f}: {', '.join(r['confidence']['triggered_emotions']) or '—'}
6. **Emotional state** → {r['emotional_state']['dominant_emotion']}, intensity {r['emotional_state']['intensity']:.2f}, severity {r['emotional_state']['severity']}
7. **Semantic matching** → {len(r['semantic_similarity'])} content items scored via {r['semantic_backend']}
8. **Hybrid candidates** → rule + preference + collaborative + history + ML features
9. **Ranking** → {len(r['recommendations'])} item(s) returned
""")


def render_feedback_log():
    log = st.session_state["feedback_log"]
    if not log:
        return
    st.markdown("#### 4 · Feedback logged this session")
    show_df(pd.DataFrame(log))
    st.caption("Feedback is appended to the interaction log. Click **Generate recommendations** again to "
               "retrain the personalized model on it. Reward values: "
               + ", ".join(f"{k}={v}" for k, v in REWARD_MAP.items() if k in ("helpful", "complete", "skip", "hide")))


def run_controlled_evaluation(cfg, models):
    """Run the fixed Task 9 dataset without persisting history or feedback."""
    cases = load_evaluation_cases(DEFAULT_EVAL_CASES_PATH)
    contents_by_id = {c.content_id: c for c in cfg["contents"]}

    def advanced(case):
        case_cfg = dict(cfg)
        case_cfg["user_id"] = case.get("user_id", "eval_user")
        case_cfg["top_k"] = min(cfg["top_k"], int(case.get("k", cfg["top_k"])))
        generate_m3(case["text"], case_cfg, models, persist_history=False)
        result = st.session_state.get("m3") or {}
        return [
            {"content_id": x["content_id"], "content_type": contents_by_id.get(x["content_id"], None).content_type if contents_by_id.get(x["content_id"]) else None}
            for x in result.get("recommendations", [])
        ]

    def baseline(case):
        case_cfg = dict(cfg)
        case_cfg["user_id"] = case.get("user_id", "eval_user")
        case_cfg["top_k"] = min(cfg["top_k"], int(case.get("k", cfg["top_k"])))
        generate_m3(case["text"], case_cfg, models, persist_history=False)
        result = st.session_state.get("m3") or {}
        rows = sorted(result.get("candidates", []), key=lambda x: (-x["rule_score"], x["content_id"]))[:case_cfg["top_k"]]
        return [
            {"content_id": x["content_id"], "content_type": contents_by_id.get(x["content_id"], None).content_type if contents_by_id.get(x["content_id"]) else None}
            for x in rows
        ]

    return run_task9_evaluation(cases, baseline, advanced)


def render_m3_part2_dashboard(r, cfg, models):
    """Milestone 3 Part 2: Tasks 7–10 UI."""
    st.markdown("### 🧪 Milestone 3 Part 2 — Learning, Explainability & Evaluation")

    feedback_records = load_feedback(cfg["feedback_path"], cfg["user_id"])
    stats = feedback_summary(feedback_records)

    st.markdown("#### Task 7 · Feedback capture & learning")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Feedback events", stats["total"])
    f2.metric("Average rating", "—" if stats["average_rating"] is None else stats["average_rating"])
    f3.metric("Acceptance rate", f"{stats['acceptance_rate']:.1%}")
    f4.metric("Feedback model", "Fitted" if r["personalization"].get("feedback_model_fitted") else "Baseline")
    if stats["event_counts"]:
        show_df(pd.DataFrame(
            [{"event": k, "count": v} for k, v in sorted(stats["event_counts"].items())]
        ))
    else:
        st.info("Use Helpful / Completed / Skip / Hide above to create feedback data.")

    st.markdown("#### Task 8 · Dynamic recommendation explanations")
    for rec in r["recommendations"]:
        exp = rec.get("explanation", {})
        if exp:
            with st.expander(f"{rec['title']} — explanation"):
                st.write(" · ".join(exp.get("reasons", [])))
                show_df(pd.DataFrame([exp.get("evidence", {})]))

    st.markdown("#### Task 9 · Recommendation evaluation")
    st.caption("Live feedback is shown separately. Controlled Task 9 acceptance is measured only from accepted_ids in the fixed evaluation dataset.")
    relevant_ids = {
        x.content_id for x in feedback_records
        if x.event in {"accept", "complete", "helpful"}
    }
    predicted = [x["content_id"] for x in r["recommendations"]]
    if relevant_ids:
        ev = evaluate_rankings(
            "advanced_ml", [predicted], [relevant_ids],
            k=cfg["top_k"],
            accepted=[x in relevant_ids for x in predicted],
            diversities=[recommendation_diversity(r["recommendations"])],
        ).to_dict()
        show_df(pd.DataFrame([ev]))

        baseline_order = [
            x["content_id"] for x in sorted(
                r["candidates"],
                key=lambda x: (-x["rule_score"], x["content_id"])
            )[:cfg["top_k"]]
        ]
        baseline_div = recommendation_diversity([
            {"content_id": cid, "content_type": r["_contents_by_id"][cid].content_type}
            for cid in baseline_order
        ])
        baseline = evaluate_rankings(
            "baseline", [baseline_order], [relevant_ids],
            k=cfg["top_k"], diversities=[baseline_div]
        ).to_dict()
        advanced = evaluate_rankings(
            "advanced_ml", [predicted], [relevant_ids],
            k=cfg["top_k"], diversities=[recommendation_diversity(r["recommendations"])]
        ).to_dict()
        comparison = compare_baseline_and_advanced({"baseline": baseline, "advanced_ml": advanced})
        st.write("**Baseline vs advanced-ML deltas**")
        show_df(pd.DataFrame([comparison["delta_advanced_minus_baseline"]]))
        st.caption(comparison["performance_note"])
    else:
        st.info("No accepted/completed/helpful feedback yet, so feedback-based quality metrics are not available.")

    st.markdown("#### Task 9 · Controlled evaluation dataset")
    st.caption(f"The controlled dataset is stored in {DEFAULT_EVAL_CASES_PATH} and keeps baseline vs advanced testing reproducible.")
    if os.path.exists(DEFAULT_EVAL_CASES_PATH):
        try:
            cases = json.load(open(DEFAULT_EVAL_CASES_PATH, "r", encoding="utf-8"))
            show_df(pd.DataFrame(cases))
            st.info(f"Loaded {len(cases)} controlled evaluation cases.")
            if st.button("Run controlled baseline vs advanced evaluation", key="run_controlled_eval"):
                try:
                    with st.spinner("Evaluating all controlled cases..."):
                        comparison = run_controlled_evaluation(cfg, models)
                    st.success("Controlled evaluation completed.")
                    show_df(pd.DataFrame([comparison["baseline"], comparison["advanced_ml"]]))
                    st.write("Metric deltas (advanced − baseline)")
                    show_df(pd.DataFrame([comparison["delta_advanced_minus_baseline"]]))
                    st.caption(comparison["performance_note"])
                except Exception as e:
                    logger.exception("Controlled evaluation failed")
                    st.error(f"Controlled evaluation failed: {e}")
        except Exception as e:
            st.warning(f"Could not read controlled evaluation dataset: {e}")

    st.markdown("#### Task 10 · Regression validation")
    if st.button("🧪 Run Milestone 3 Part 2 regression checks", key="btn_m3_regression"):
        try:
            with st.spinner("Running regression checks..."):
                result = run_regression_tests()
            st.success(f"Regression suite passed: {result['checks']} checks.")
            show_df(pd.DataFrame([result]))
        except Exception as e:
            logger.exception("Regression checks failed")
            st.error(f"Regression checks failed: {e}")

    with st.expander("📁 Part 2 data files"):
        st.write(f"Emotion history: `{cfg['history_path']}`")
        st.write(f"Recommendation feedback: `{cfg['feedback_path']}`")


def render_milestone3(a, cfg, models):
    st.subheader("🌿 Personalized Wellness Recommendations")

    if not cfg["run_recs"]:
        st.info("Milestone 3 is switched off. Tick **Run Milestone 3 recommendations** in the sidebar.")
        return
    if not a["valid_rows"]:
        st.info("No valid rows to generate recommendations for.")
        return

    rows = a["valid_rows"]
    if len(rows) > 1:
        idx = st.selectbox(
            "Choose a record to get recommendations for",
            options=list(range(len(rows))),
            format_func=lambda i: f"#{rows[i][0]} · " + (rows[i][1][:80] + ("…" if len(rows[i][1]) > 80 else "")),
            key="m3_record_idx",
        )
    else:
        idx = 0
    text = rows[idx][1]

    if st.button("🌿 Generate recommendations", type="primary", key="btn_generate_m3"):
        try:
            with st.spinner("Running emotional-state analysis, semantic matching and hybrid ranking..."):
                generate_m3(text, cfg, models)
        except Exception as e:
            logger.exception("Milestone 3 generation failed")
            st.session_state["m3"] = None
            st.error(f"Recommendation stage failed: {e}")

    r = st.session_state["m3"]
    if not r:
        st.caption("Click the button to run Milestone 3 on the selected text.")
        return
    if not r.get("is_valid"):
        st.error(f"Input rejected: {r.get('error')}")
        return

    if r["signature"] != m3_signature(text, cfg):
        st.warning("Inputs or settings changed since these recommendations were generated — "
                   "click **Generate recommendations** to refresh.")

    st.caption(f"Text: “{r['input_text'][:200]}{'…' if len(r['input_text']) > 200 else ''}”")

    render_emotional_state(r)
    st.divider()
    render_recommendations(r)
    st.divider()
    render_ranking_details(r)
    render_feedback_log()
    st.divider()
    render_m3_part2_dashboard(r, cfg, models)

    export = {k: v for k, v in r.items() if not k.startswith("_") and k not in ("signature",)}
    st.download_button(
        "⬇️ Download Milestone 3 result (JSON)",
        json.dumps(export, indent=2, default=str),
        file_name="mood_mentor_recommendations.json",
        mime="application/json",
        key="dl_m3_json",
    )


# ===========================================================
# MILESTONE 4 -- DASHBOARD / REPORTS / SEARCH / PRIVACY
# ===========================================================

def render_milestone4(cfg):
    st.header("📊 Milestone 4 — Dashboard, History, Reports & Validation")
    try:
        validate_user_id(cfg["user_id"])
    except ValueError as exc:
        st.error(str(exc))
        return

    snap = dashboard_snapshot(cfg["history_path"], DEFAULT_DB_PATH, cfg["user_id"])
    history = snap["history"]

    # Task 1/2: dashboard + trend visualization
    st.subheader("Task 1–2 · Emotional dashboard & trends")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emotion records", snap["history_samples"])
    c2.metric("Recommendations", snap["recommendation_samples"])
    c3.metric("Feedback events", snap["feedback_samples"])
    c4.metric("Dominant emotion", history["dominant_emotion"].iloc[-1] if not history.empty and "dominant_emotion" in history else "—")

    if not history.empty:
        granularity = st.selectbox("Trend granularity", ["Daily", "Weekly", "Monthly"], key="m4_granularity")
        freq = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[granularity]
        trends = aggregate_emotion_trends(history, freq)
        if not trends.empty:
            chart = trends.set_index("period")[["avg_intensity", "avg_polarity"]]
            st.line_chart(chart)
        freq_df = snap["emotion_frequency"]
        if not freq_df.empty:
            st.bar_chart(freq_df.set_index("emotion"))
    else:
        st.info("No stored emotion history yet. Run an analysis with history persistence enabled.")

    # Task 4: search/filter
    st.subheader("Task 4 · Search & filtering")
    q = sanitize_search_query(st.text_input("Search history/recommendations/feedback"))
    emotions = sorted(history["dominant_emotion"].dropna().astype(str).unique()) if not history.empty and "dominant_emotion" in history else []
    selected_emotions = st.multiselect("Emotion filter", emotions)
    min_i, max_i = st.slider("Intensity range", 0.0, 1.0, (0.0, 1.0), 0.05)
    filtered = filter_records(history, emotions=selected_emotions, min_intensity=min_i, max_intensity=max_i)
    filtered = search_dataframe(filtered, q)
    if not filtered.empty:
        show_df(filtered)
    else:
        st.caption("No records match the selected filters.")

    # Task 3: recommendation history + feedback
    st.subheader("Task 3 · Recommendation history & feedback")
    recs = enrich_recommendation_history(snap["recommendations"])
    recs = search_dataframe(recs, q)
    feedback = search_dataframe(snap["feedback"], q)
    if not recs.empty and "content_type" in recs.columns:
        rec_types = sorted(recs["content_type"].dropna().astype(str).unique())
        selected_types = st.multiselect("Recommendation type", rec_types)
        if selected_types:
            recs = recs[recs["content_type"].isin(selected_types)]
    if not feedback.empty and "event" in feedback.columns:
        feedback_events = sorted(feedback["event"].dropna().astype(str).unique())
        selected_events = st.multiselect("Feedback status/event", feedback_events)
        if selected_events:
            feedback = feedback[feedback["event"].isin(selected_events)]
    rt, ft = st.tabs(["Recommendation history", "Feedback"])
    with rt:
        show_df(recs)
    with ft:
        show_df(feedback)

    # Task 5: exports
    st.subheader("Task 5 · Reports & export")
    bundle = build_csv_bundle(snap)
    for name, data in bundle.items():
        st.download_button(f"Download {name}", data, file_name=name, mime="text/csv", key=f"m4_{name}")
    if st.button("Generate PDF report", key="m4_pdf"):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        pdf_path = os.path.join(REPORTS_DIR, f"mood_mentor_{cfg['user_id']}.pdf")
        try:
            generate_pdf_report(snap, pdf_path)
            with open(pdf_path, "rb") as fh:
                st.download_button("Download PDF", fh.read(), file_name=os.path.basename(pdf_path), mime="application/pdf", key="m4_pdf_download")
        except Exception as exc:
            logger.exception("PDF report generation failed")
            st.error(f"PDF generation failed: {exc}")

    # Task 7: performance
    st.subheader("Task 7 · Model/recommendation performance smoke test")
    if st.button("Run recommendation stress test", key="m4_stress"):
        from src.stress_test import run_recommendation_stress
        st.json(run_recommendation_stress(50))

    # Task 8: privacy / deletion
    st.subheader("Task 8 · Privacy & data deletion")
    st.caption("All dashboard searches are local. Delete removes this user's persisted SQLite records.")
    if st.checkbox("I understand that this permanently deletes this user's stored database records.", key="m4_delete_confirm"):
        if st.button("Delete my stored data", type="secondary", key="m4_delete"):
            from src.database import delete_user_data
            try:
                deleted = delete_user_data(DEFAULT_DB_PATH, cfg["user_id"])
                st.success(f"Deleted records: {deleted}")
            except Exception as exc:
                logger.exception("Data deletion failed")
                st.error(f"Deletion failed: {exc}")

    # Task 10: validation
    st.subheader("Task 10 · Final validation")
    if st.button("Run all automated tests", key="m4_tests"):
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("All automated tests passed.")
        else:
            st.error("Some automated tests failed.")
        st.code((result.stdout + "\n" + result.stderr)[-12000:])


# ===========================================================
# SIDEBAR
# ===========================================================

def render_sidebar():
    """Renders every sidebar control and returns (mode, cfg, run_emotion,
    bert_dir, distil_dir, max_emotion_rows)."""
    st.sidebar.title("Mood Mentor")

    mode = st.sidebar.radio("Choose input method", [MODE_CHAT, MODE_TXT, MODE_CSV])

    # ---- Milestone 2 settings ----
    st.sidebar.divider()
    st.sidebar.subheader("Milestone 2 — Emotion Models")

    run_emotion = st.sidebar.checkbox("Run BERT + DistilBERT emotion analysis", value=True)
    bert_dir = st.sidebar.text_input("BERT model folder", value=DEFAULT_BERT_DIR)
    distil_dir = st.sidebar.text_input("DistilBERT model folder", value=DEFAULT_DISTIL_DIR)
    max_emotion_rows = st.sidebar.number_input(
        "Max rows to run through emotion models (TXT/CSV modes)",
        min_value=1, max_value=200, value=20, step=1,
        help="Transformer inference is slower than VADER, so batch modes cap how many valid "
             "rows go through Milestone 2. Raise this if you need the full file analyzed.",
    )

    # ---- Milestone 3 settings ----
    st.sidebar.divider()
    st.sidebar.subheader("Milestone 3 — Recommendations")

    run_recs = st.sidebar.checkbox("Run Milestone 3 recommendations", value=True)
    m3_model_name = st.sidebar.selectbox(
        "Emotion model feeding Milestone 3", ["BERT", "DistilBERT"],
        help="pipeline_v3 uses BERT. DistilBERT produces the same score format and is faster.",
    )
    top_k = st.sidebar.slider("Top-K recommendations", 1, 10, 5)
    threshold = st.sidebar.slider(
        "Low-relevance cutoff", 0.0, 0.6, 0.20, 0.01,
        help="Items scoring below this are dropped by the ranker (default 0.20).",
    )
    semantic_model_name = st.sidebar.text_input("Semantic model", value=DEFAULT_SEMANTIC_MODEL)

    # wellness content library
    content_source = st.sidebar.radio("Wellness content library", ["Built-in (7 items)", "Upload CSV"])
    contents = DEFAULT_WELLNESS_CONTENT
    content_key = "builtin"
    if content_source == "Upload CSV":
        up = st.sidebar.file_uploader("Wellness content CSV", type=["csv"], key="content_csv")
        st.sidebar.download_button(
            "Download CSV template", content_template_csv(),
            file_name="wellness_content_template.csv", mime="text/csv", key="dl_content_template",
        )
        if up is not None:
            try:
                parsed = parse_content_csv(up.getvalue())
                if parsed:
                    contents, content_key = parsed, f"upload:{up.name}:{up.size}"
                else:
                    st.sidebar.warning("That CSV has no rows — using the built-in library.")
            except Exception as e:
                logger.warning("Could not read uploaded content CSV: %s", e)
                st.sidebar.error(f"Couldn't read content CSV ({e}) — using the built-in library.")

    # user profile
    st.sidebar.markdown("**User profile**")
    user_id = st.sidebar.text_input("User ID", value="demo_user")
    type_options = sorted({c.content_type for c in contents})
    tag_options = sorted({t for c in contents for t in c.tags})
    by_id_all = {c.content_id: c for c in contents}
    preferred_types = st.sidebar.multiselect("Preferred content types", type_options, key=f"pt_{content_key}")
    preferred_tags = st.sidebar.multiselect("Preferred tags", tag_options, key=f"ptag_{content_key}")
    blocked = st.sidebar.multiselect(
        "Blocked content", list(by_id_all), format_func=lambda cid: f"{cid} — {by_id_all[cid].title}",
        key=f"blk_{content_key}",
    )

    # interaction history
    st.sidebar.markdown("**Interaction history**")
    use_history = st.sidebar.checkbox("Use interaction history", value=True)
    interactions_path = st.sidebar.text_input("Interaction log (CSV)", value=DEFAULT_INTERACTIONS_PATH)
    feedback_path = st.sidebar.text_input("Part 2 feedback log (CSV)", value=DEFAULT_FEEDBACK_PATH)
    history_path = st.sidebar.text_input("Emotion history log (CSV)", value=DEFAULT_HISTORY_PATH)
    if st.sidebar.button("Generate demo interaction history"):
        try:
            n = seed_demo_interactions(interactions_path, contents)
            st.sidebar.success(f"Appended {n} synthetic interactions (demo_user_1…6) to {interactions_path}.")
        except Exception as e:
            logger.exception("Could not write demo interaction history")
            st.sidebar.error(f"Couldn't write demo history: {e}")
    st.sidebar.caption("Demo history is synthetic and only meant to show collaborative filtering and the "
                       "Random-Forest personalization at work. Delete the CSV to reset.")

    cfg = {
        "run_recs": run_recs, "m3_model_name": m3_model_name, "top_k": top_k, "threshold": threshold,
        "semantic_model_name": semantic_model_name.strip() or DEFAULT_SEMANTIC_MODEL,
        "contents": contents, "content_key": content_key,
        "user_id": user_id.strip() or "demo_user",
        "preferred_types": preferred_types, "preferred_tags": preferred_tags, "blocked": blocked,
        "use_history": use_history,
        "interactions_path": interactions_path.strip() or DEFAULT_INTERACTIONS_PATH,
        "feedback_path": feedback_path.strip() or DEFAULT_FEEDBACK_PATH,
        "history_path": history_path.strip() or DEFAULT_HISTORY_PATH,
    }

    return mode, cfg, run_emotion, bert_dir, distil_dir, max_emotion_rows


# ===========================================================
# MAIN
# ===========================================================

def main():
    st.set_page_config(page_title="Mood Mentor", page_icon="🧠", layout="wide")
    init_session_state()

    st.title("🧠 Mood Mentor")
    st.subheader("AI Emotion Understanding & Personalized Wellness")
    st.write(
        "Analyze text through the full pipeline: preprocessing → VADER sentiment → "
        "BERT / DistilBERT emotion classification → emotional state → semantic matching → "
        "hybrid recommendation & ranking → feedback-driven personalization."
    )

    mode, cfg, run_emotion, bert_dir, distil_dir, max_emotion_rows = render_sidebar()

    # ---- load models once (only if some stage needs them) ----
    load_errors = {}
    bert_model = bert_tok = distil_model = distil_tok = None
    if run_emotion or cfg["run_recs"]:
        with st.spinner("Loading emotion models (first run only)..."):
            bert_model, bert_tok, distil_model, distil_tok, load_errors = load_emotion_models(bert_dir, distil_dir)
    models = (bert_model, bert_tok, distil_model, distil_tok)

    if cfg["run_recs"]:
        for name, msg in load_errors.items():
            if name == cfg["m3_model_name"]:
                st.sidebar.error(f"{name} not loaded — Milestone 3 needs it. ({msg})")

    def do_analysis(the_mode, text=None, uploaded=None):
        """Milestone 1 (+2) run; result kept in session_state so it survives the
        reruns triggered by the feedback buttons and sidebar changes."""
        df, summary = ingest_and_analyze(the_mode, text=text, uploaded=uploaded)

        valid_rows = [(row["id"], row["input_text"]) for _, row in df.iterrows() if row["is_valid"]]

        emotion_results, capped = [], False
        if run_emotion:
            rows = valid_rows if the_mode == MODE_CHAT else valid_rows[: int(max_emotion_rows)]
            capped = len(rows) < len(valid_rows)
            emotion_results = run_emotion_stage(rows, bert_model, bert_tok, distil_model, distil_tok)

        st.session_state["analysis"] = {
            "mode": the_mode, "df": df, "summary": summary, "valid_rows": valid_rows,
            "emotion_results": emotion_results, "emotion_capped": capped,
            "ran_emotion": run_emotion, "load_errors": dict(load_errors),
        }
        st.session_state["m3"] = None

        # Chat mode is a single text -> run the whole pipeline (Milestone 3 included) in one click.
        if the_mode == MODE_CHAT and cfg["run_recs"] and valid_rows:
            try:
                generate_m3(valid_rows[0][1], cfg, models)
            except Exception as e:
                logger.exception("Milestone 3 generation failed during chat-mode analysis")
                st.error(f"Recommendation stage failed: {e}")

    # ---- INPUT SECTIONS ----
    if mode == MODE_CHAT:
        st.header("💬 Analyze Text")

        st.caption("Try an example:")
        ex_cols = st.columns(len(EXAMPLES))
        for col, (label, sample) in zip(ex_cols, EXAMPLES.items()):
            with col:
                st.button(label, key=f"ex_{label}", on_click=_set_chat_text, args=(sample,))

        text = st.text_area(
            "Enter your message or journal entry:",
            height=200,
            key="chat_text",
            placeholder="Example: I have been feeling really anxious about work today...",
        )

        if st.button("🔍 Analyze Mood", type="primary"):
            if not text.strip():
                st.warning("Please enter some text first.")
            else:
                try:
                    with st.spinner("Analyzing your mood..."):
                        do_analysis(MODE_CHAT, text=text)
                except Exception as e:
                    logger.exception("Milestone 1/2 analysis failed (chat mode)")
                    st.session_state["analysis"] = None
                    st.error(f"Something went wrong during analysis: {e}")

    elif mode == MODE_TXT:
        st.header("📄 Upload Journal / Text File")
        uploaded_txt = st.file_uploader("Upload a .txt file", type=["txt"])

        if uploaded_txt is not None:
            st.info(f"Uploaded: {uploaded_txt.name}")
            if st.button("🔍 Analyze File", type="primary"):
                try:
                    with st.spinner("Analyzing your file..."):
                        do_analysis(MODE_TXT, uploaded=uploaded_txt)
                except Exception as e:
                    logger.exception("Milestone 1/2 analysis failed (txt mode)")
                    st.session_state["analysis"] = None
                    st.error(f"Unable to process the file: {e}")

    elif mode == MODE_CSV:
        st.header("📊 Upload CSV Mood Data")
        uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_csv is not None:
            try:
                preview_df = pd.read_csv(uploaded_csv)
                st.write("### File Preview")
                show_df(preview_df.head())
                st.write(f"Rows: {len(preview_df)}")
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
            finally:
                uploaded_csv.seek(0)

            if st.button("🔍 Analyze CSV", type="primary"):
                try:
                    with st.spinner("Analyzing CSV..."):
                        do_analysis(MODE_CSV, uploaded=uploaded_csv)
                except Exception as e:
                    logger.exception("Milestone 1/2 analysis failed (csv mode)")
                    st.session_state["analysis"] = None
                    st.error(f"Unable to analyze CSV: {e}")

    # ---- RESULTS -- one tab per milestone ----
    analysis = st.session_state["analysis"]
    if analysis and analysis["mode"] == mode:
        st.success("Analysis completed!")

        tab1, tab2, tab3, tab4 = st.tabs([
            "1 · Sentiment (VADER)",
            "2 · Emotions (BERT / DistilBERT)",
            "3 · Recommendations",
            "4 · Dashboard / Reports",
        ])
        with tab1:
            render_milestone1(analysis)
        with tab2:
            render_milestone2(analysis)
        with tab3:
            render_milestone3(analysis, cfg, models)
        with tab4:
            render_milestone4(cfg)

    # ---- FOOTER ----
    st.divider()
    st.caption(
        "Mood Mentor — Milestone 1 (Ingestion + Preprocessing + VADER) "
        "+ Milestone 2 (BERT / DistilBERT Emotion Classification + Confidence Scores) "
        "+ Milestone 3 (Emotional State, Semantic Matching, Hybrid Recommendations & Personalization). "
        "Milestone 4 adds dashboard trends, history/search/filtering, exports, privacy deletion, stress testing "
        "and packaging support. Wellness suggestions are general guidance, not medical advice."
    )


if __name__ == "__main__":
    main()