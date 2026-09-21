"""
app.py
Mood Mentor - Streamlit UI (Milestones 1, 2 and 3)

One app that walks a piece of text through the whole pipeline:

    Enter Text -> Text Preprocessing -> VADER Sentiment            (Milestone 1)
    -> BERT / DistilBERT Emotion Classification -> Confidence      (Milestone 2)
    -> Emotional State Analysis -> Semantic Matching
    -> Hybrid Recommendation (rules + collaborative + ML) -> Ranking
    -> User Feedback -> Personalization                            (Milestone 3)

Nothing in src/ is modified. Milestone 3 is wired stage-by-stage (the same
order as src/pipeline_v3.py) instead of calling run_milestone3() directly, so
that the BERT / DistilBERT models and the sentence-transformer are loaded ONCE
and cached, rather than reloaded on every click, and so every intermediate
result can be shown in the UI.

Run:  streamlit run app.py
"""

import json
import os
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


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mood Mentor",
    page_icon="🧠",
    layout="wide",
)

MODE_CHAT = "💬 Chat Text"
MODE_TXT = "📄 TXT File"
MODE_CSV = "📊 CSV File"

DEFAULT_INTERACTIONS_PATH = os.path.join("data", "interactions.csv")
DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"

EXAMPLES = {
    "😰 Excited but nervous": "I am excited about the new opportunity but nervous about the outcome.",
    "😔 Low & overwhelmed": "I have been feeling really down and overwhelmed at work lately, nothing seems to go right.",
    "😠 Frustrated": "I'm furious that my manager took credit for my work again. I can't stop thinking about it.",
}

for _key, _default in {"analysis": None, "m3": None, "feedback_log": [], "chat_text": ""}.items():
    st.session_state.setdefault(_key, _default)


# ---------------------------------------------------------
# SMALL UI HELPERS
# ---------------------------------------------------------

def show_df(df, **kwargs):
    """st.dataframe at full width, across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, **kwargs)


def _set_chat_text(value: str):
    st.session_state["chat_text"] = value


# ---------------------------------------------------------
# MODEL / RESOURCE LOADING (cached -- runs once per session/path)
# ---------------------------------------------------------

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
        errors["BERT"] = str(e)

    try:
        distil_model, distil_tok = emotion_distilbert.load_trained_model(distil_dir)
    except Exception as e:
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


# ---------------------------------------------------------
# MILESTONE 1 + 2 -- ANALYSIS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# MILESTONE 3 -- RECOMMENDATION STAGE
# ---------------------------------------------------------

def m3_signature(text: str, cfg: dict) -> str:
    """Identifies the inputs + settings a recommendation result was built from,
    so the UI can tell when it has gone stale."""
    return json.dumps([
        text, cfg["m3_model_name"], cfg["semantic_model_name"], cfg["top_k"], cfg["threshold"],
        cfg["user_id"], sorted(cfg["preferred_types"]), sorted(cfg["preferred_tags"]),
        sorted(cfg["blocked"]), cfg["content_key"], cfg["use_history"], cfg["interactions_path"],
    ])


def generate_m3(text: str, cfg: dict, models: tuple):
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

    contents = cfg["contents"]
    matcher, backend, backend_error = load_semantic_matcher(cfg["semantic_model_name"])
    matcher.fit(contents)  # re-encode so a changed content library is never scored against stale vectors
    semantic_scores = matcher.match_emotional_state(state, record.raw_text, contents)

    interactions = load_interactions(cfg["interactions_path"]) if cfg["use_history"] else []
    personalized_model = train_personalized_model(interactions)
    usable_for_ml = sum(1 for x in interactions if len(x.feature_vector) == len(FEATURE_NAMES))

    profile = UserProfile(
        user_id=cfg["user_id"],
        preferred_types=list(cfg["preferred_types"]),
        preferred_tags=list(cfg["preferred_tags"]),
        blocked_content_ids=set(cfg["blocked"]),
    )

    engine = HybridRecommendationEngine(contents, interactions, personalized_model)
    candidates = engine.generate_candidates(state, profile, semantic_scores)

    ranker = RecommendationRanker(low_relevance_threshold=cfg["threshold"])
    ranked = ranker.rank(candidates, state, top_k=cfg["top_k"])

    candidate_rows = []
    for row in candidates:
        candidate_rows.append({
            "content_id": row["content"].content_id,
            "title": row["content"].title,
            "final_score": round(ranker.score_row(row, state), 4),
            "personalized_ml": round(row["personalized_ml_score"], 4),
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
        "semantic_backend": backend,
        "semantic_backend_error": backend_error,
        "semantic_similarity": semantic_scores,
        "recommendations": [r.to_dict() for r in ranked],
        "candidates": candidate_rows,
        "features": {row["content"].content_id: feature_dict(row["features"]) for row in candidates},
        "ranking_weights": ranked[0].components["ranking_weights"] if ranked else ranker._weights(state.intensity),
        "personalization": {
            "n_interactions": len(interactions),
            "usable_for_ml": usable_for_ml,
            "ml_fitted": bool(personalized_model.fitted),
            "baseline": personalized_model.baseline,
        },
        "user_id": cfg["user_id"],
        "interactions_path": cfg["interactions_path"],
        "threshold": cfg["threshold"],
        "_contents_by_id": {c.content_id: c for c in contents},
    }


def log_feedback(content_id: str, interaction_type: str):
    """on_click handler for the feedback buttons -> src/interaction_service.py"""
    r = st.session_state.get("m3")
    if not r or not r.get("is_valid"):
        return
    try:
        interaction = record_interaction(
            r["interactions_path"], r["user_id"], content_id, interaction_type,
            emotion_scores=r["emotional_state"]["emotion_probabilities"],
            emotion_intensity=r["emotional_state"]["intensity"],
            feature_vector=r["features"].get(content_id, {}),
        )
        st.session_state["feedback_log"].append({
            "user_id": interaction.user_id,
            "content_id": interaction.content_id,
            "interaction": interaction.interaction_type,
            "reward": interaction.reward,
            "timestamp": interaction.timestamp,
        })
        st.toast(f"Logged “{interaction_type}” for {content_id} (reward {interaction.reward:.1f})")
    except Exception as e:
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


# ---------------------------------------------------------
# RENDERING -- MILESTONE 1
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# RENDERING -- MILESTONE 2
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# RENDERING -- MILESTONE 3
# ---------------------------------------------------------

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
                st.write("**Why:** " + " · ".join(rec["reasons"]))
            with score_col:
                st.metric("Hybrid score", f"{rec['score']:.3f}")

            with st.expander("Score breakdown"):
                comps = {k: v for k, v in rec["components"].items()
                         if k not in ("ranking_weights", "emotional_intensity")}
                st.bar_chart(pd.Series(comps, name="component score"))

            st.caption("Was this useful? Your feedback is logged and used for personalization.")
            b1, b2, b3, b4 = st.columns(4)
            for col, (label, kind) in zip(
                (b1, b2, b3, b4),
                (("👍 Helpful", "helpful"), ("✅ Completed", "complete"),
                 ("⏭️ Skip", "skip"), ("🙈 Hide", "hide")),
            ):
                with col:
                    st.button(
                        label, key=f"fb_{rec['rank']}_{rec['content_id']}_{kind}",
                        on_click=log_feedback, args=(rec["content_id"], kind),
                    )


def render_ranking_details(r):
    st.markdown("#### 3 · How the ranking was built")

    p = r["personalization"]
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

    export = {k: v for k, v in r.items() if not k.startswith("_") and k not in ("signature",)}
    st.download_button(
        "⬇️ Download Milestone 3 result (JSON)",
        json.dumps(export, indent=2, default=str),
        file_name="mood_mentor_recommendations.json",
        mime="application/json",
        key="dl_m3_json",
    )


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🧠 Mood Mentor")
st.subheader("AI Emotion Understanding & Personalized Wellness")

st.write(
    "Analyze text through the full pipeline: preprocessing → VADER sentiment → "
    "BERT / DistilBERT emotion classification → emotional state → semantic matching → "
    "hybrid recommendation & ranking → feedback-driven personalization."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Mood Mentor")

mode = st.sidebar.radio("Choose input method", [MODE_CHAT, MODE_TXT, MODE_CSV])

# ---- Milestone 2 settings ----
st.sidebar.divider()
st.sidebar.subheader("Milestone 2 — Emotion Models")

run_emotion = st.sidebar.checkbox("Run BERT + DistilBERT emotion analysis", value=True)
bert_dir = st.sidebar.text_input("BERT model folder", value=emotion_bert.DEFAULT_SAVE_DIR)
distil_dir = st.sidebar.text_input("DistilBERT model folder", value=emotion_distilbert.DEFAULT_SAVE_DIR)
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
if st.sidebar.button("Generate demo interaction history"):
    try:
        n = seed_demo_interactions(interactions_path, contents)
        st.sidebar.success(f"Appended {n} synthetic interactions (demo_user_1…6) to {interactions_path}.")
    except Exception as e:
        st.sidebar.error(f"Couldn't write demo history: {e}")
st.sidebar.caption("Demo history is synthetic and only meant to show collaborative filtering and the "
                   "Random-Forest personalization at work. Delete the CSV to reset.")

cfg = {
    "run_recs": run_recs, "m3_model_name": m3_model_name, "top_k": top_k, "threshold": threshold,
    "semantic_model_name": semantic_model_name.strip() or DEFAULT_SEMANTIC_MODEL,
    "contents": contents, "content_key": content_key,
    "user_id": user_id.strip() or "demo_user",
    "preferred_types": preferred_types, "preferred_tags": preferred_tags, "blocked": blocked,
    "use_history": use_history, "interactions_path": interactions_path.strip() or DEFAULT_INTERACTIONS_PATH,
}

# ---- load models once (only if some stage needs them) ----
bert_model = bert_tok = distil_model = distil_tok = None
load_errors = {}
if run_emotion or run_recs:
    with st.spinner("Loading emotion models (first run only)..."):
        bert_model, bert_tok, distil_model, distil_tok, load_errors = load_emotion_models(bert_dir, distil_dir)
models = (bert_model, bert_tok, distil_model, distil_tok)

if run_recs:
    for _name, _msg in load_errors.items():
        if _name == m3_model_name:
            st.sidebar.error(f"{_name} not loaded — Milestone 3 needs it. ({_msg})")


# ---------------------------------------------------------
# ANALYSIS RUNNER
# ---------------------------------------------------------

def do_analysis(the_mode, text=None, uploaded=None):
    """Milestone 1 (+2) run; result kept in session_state so it survives the
    reruns triggered by the feedback buttons and sidebar changes."""
    df, summary = ingest_and_analyze(the_mode, text=text, uploaded=uploaded)

    valid_rows = [(r["id"], r["input_text"]) for _, r in df.iterrows() if r["is_valid"]]

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
    if the_mode == MODE_CHAT and run_recs and valid_rows:
        try:
            generate_m3(valid_rows[0][1], cfg, models)
        except Exception as e:
            st.error(f"Recommendation stage failed: {e}")


# ---------------------------------------------------------
# INPUT SECTIONS
# ---------------------------------------------------------

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
                st.session_state["analysis"] = None
                st.error(f"Unable to analyze CSV: {e}")


# ---------------------------------------------------------
# RESULTS -- one tab per milestone
# ---------------------------------------------------------

analysis = st.session_state["analysis"]

if analysis and analysis["mode"] == mode:
    st.success("Analysis completed!")

    tab1, tab2, tab3 = st.tabs([
        "1 · Sentiment (VADER)",
        "2 · Emotions (BERT / DistilBERT)",
        "3 · Recommendations",
    ])
    with tab1:
        render_milestone1(analysis)
    with tab2:
        render_milestone2(analysis)
    with tab3:
        render_milestone3(analysis, cfg, models)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Mood Mentor — Milestone 1 (Ingestion + Preprocessing + VADER) "
    "+ Milestone 2 (BERT / DistilBERT Emotion Classification + Confidence Scores) "
    "+ Milestone 3 (Emotional State, Semantic Matching, Hybrid Recommendations & Personalization). "
    "Wellness suggestions are general guidance, not medical advice."
)