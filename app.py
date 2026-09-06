"""
app.py
Mood Mentor - Streamlit UI

Wires the UI straight to the existing pipeline, following the exact flow
from the milestone spec:

    Enter Text -> Text Preprocessing -> VADER Sentiment Analysis
    -> BERT / DistilBERT Emotion Analysis -> Emotion Classification
    -> Confidence Scores -> Final Analysis Result

Milestone 1 (ingestion + preprocessing + VADER) runs through src/pipeline.py,
unchanged from before. Milestone 2 (BERT + DistilBERT emotion classification
with confidence scores) is the new stage added below, using
src/emotion_bert.py, src/emotion_distilbert.py and src/confidence.py exactly
as they're already written -- nothing in src/ was modified for this UI.
"""

import streamlit as st
import pandas as pd
import tempfile
import os

from src.pipeline import run_pipeline
from src import emotion_bert
from src import emotion_distilbert
from src.emotion_dataset import EMOTIONS


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mood Mentor",
    page_icon="🧠",
    layout="wide"
)


# ---------------------------------------------------------
# MILESTONE 2 -- MODEL LOADING (cached so this only runs once per session)
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_emotion_models(bert_dir: str, distil_dir: str):
    """Loads both fine-tuned models once. Returns (bert_model, bert_tok,
    distil_model, distil_tok, errors) -- a model that fails to load stays
    None and its problem is recorded in `errors`, instead of crashing the
    whole app (same "don't let one bad stage kill the run" philosophy as
    pipeline.py's per-record error handling)."""
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


def run_emotion_stage(rows, bert_model, bert_tok, distil_model, distil_tok):
    """
    rows: list of (id, text) for VALID records only.
    Runs Milestone 2 Task 1/2 (BERT + DistilBERT) -> Task 3 (multi-label) ->
    Task 4 (confidence scores) for each row, using compare_predictions()
    when both models are available (that's Milestone 2 Task 2's "Compare
    BERT and DistilBERT Results", already implemented in
    src/emotion_distilbert.py) and falling back to a single model if only
    one is loaded.
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
            # Same rule as pipeline.py: surface the error on this row, don't
            # take down the batch.
            entry["error"] = f"Emotion stage error: {e}"
        results.append(entry)
    return results


def _model_block(label: str, pred: dict):
    st.markdown(f"**{label}**")
    st.write(f"Primary emotion: **{pred['primary_emotion']}**  ·  confidence **{pred['primary_confidence']:.3f}**")
    triggered = ", ".join(pred["triggered_emotions"]) if pred["triggered_emotions"] else "—"
    st.caption(f"Triggered emotions (≥ {pred['threshold']:.2f} threshold): {triggered}")
    st.bar_chart(pd.Series(pred["scores"], name="confidence"))


def render_emotion_results(results, load_errors):
    st.header("🎭 Milestone 2 — Emotion Classification (BERT vs DistilBERT)")

    if load_errors:
        for model_name, msg in load_errors.items():
            st.warning(
                f"**{model_name}** could not be loaded ({msg}). "
                f"Train it first with `python -m src.train_models`, or fix the "
                f"model folder path in the sidebar."
            )

    if not results:
        st.info("No valid rows to run emotion analysis on.")
        return

    both = [r for r in results if "bert" in r and "distilbert" in r]
    if both:
        agree = sum(1 for r in both if r["agree_on_primary"])
        st.metric("BERT ↔ DistilBERT primary-emotion agreement", f"{agree}/{len(both)}")

    for r in results:
        preview = r["text"] if len(r["text"]) <= 90 else r["text"][:90] + "…"
        with st.expander(f"📝 {preview}"):
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


def build_combined_report(m1_df: pd.DataFrame, emotion_results: list) -> pd.DataFrame:
    """Merges the Milestone 1 sentiment rows with the Milestone 2 emotion
    results into one downloadable report (Task 4 / Task 9's 'final report')."""
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


def show_summary_metrics(summary, df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", summary.get("sample_size", len(df)))
    with col2:
        st.metric("Average Compound", round(summary.get("avg_compound_score", 0) or 0, 3))
    with col3:
        st.metric("Positive", summary.get("positive_count", 0))
    with col4:
        st.metric("Negative", summary.get("negative_count", 0))


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🧠 Mood Mentor")
st.subheader("AI Emotion Understanding & Personalized Wellness")

st.write(
    "Analyze text through the full pipeline: preprocessing → VADER sentiment → "
    "BERT / DistilBERT emotion classification → confidence scores."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Mood Mentor")

mode = st.sidebar.radio(
    "Choose input method",
    [
        "💬 Chat Text",
        "📄 TXT File",
        "📊 CSV File"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Milestone 2 — Emotion Models")

run_emotion = st.sidebar.checkbox(
    "Run BERT + DistilBERT emotion analysis", value=True
)
bert_dir = st.sidebar.text_input("BERT model folder", value=emotion_bert.DEFAULT_SAVE_DIR)
distil_dir = st.sidebar.text_input("DistilBERT model folder", value=emotion_distilbert.DEFAULT_SAVE_DIR)
max_emotion_rows = st.sidebar.number_input(
    "Max rows to run through emotion models (TXT/CSV modes)",
    min_value=1, max_value=200, value=20, step=1,
    help="Transformer inference is slower than VADER, so batch modes cap how "
         "many valid rows go through Milestone 2 by default. Raise this if you "
         "need the full file analyzed."
)

bert_model = bert_tok = distil_model = distil_tok = None
load_errors = {}
if run_emotion:
    bert_model, bert_tok, distil_model, distil_tok, load_errors = load_emotion_models(bert_dir, distil_dir)


# ---------------------------------------------------------
# CHAT TEXT
# ---------------------------------------------------------

if mode == "💬 Chat Text":

    st.header("💬 Analyze Text")

    text = st.text_area(
        "Enter your message or journal entry:",
        height=200,
        placeholder="Example: I have been feeling really anxious about work today..."
    )

    if st.button("🔍 Analyze Mood", type="primary"):

        if not text.strip():
            st.warning("Please enter some text first.")

        else:

            with st.spinner("Analyzing your mood..."):

                try:

                    df, summary = run_pipeline("raw_text", text)

                    st.success("Analysis completed!")

                    st.header("📈 Sentiment Summary")
                    show_summary_metrics(summary, df)

                    st.header("🧠 Milestone 1 Result")
                    st.dataframe(df, use_container_width=True)

                    emotion_results = []
                    if run_emotion:
                        valid_rows = [
                            (r["id"], r["input_text"])
                            for _, r in df.iterrows() if r["is_valid"]
                        ]
                        with st.spinner("Running BERT + DistilBERT emotion analysis..."):
                            emotion_results = run_emotion_stage(
                                valid_rows, bert_model, bert_tok, distil_model, distil_tok
                            )
                        render_emotion_results(emotion_results, load_errors)

                        if emotion_results:
                            combined = build_combined_report(df, emotion_results)
                            st.download_button(
                                "⬇️ Download full report (CSV)",
                                combined.to_csv(index=False),
                                file_name="mood_mentor_report.csv",
                                mime="text/csv",
                            )

                except Exception as e:

                    st.error(
                        f"Something went wrong during analysis: {e}"
                    )


# ---------------------------------------------------------
# TXT FILE
# ---------------------------------------------------------

elif mode == "📄 TXT File":

    st.header("📄 Upload Journal / Text File")

    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"]
    )

    if uploaded_file is not None:

        st.info(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button("🔍 Analyze File", type="primary"):

            try:

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".txt"
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                with st.spinner("Analyzing your file..."):

                    df, summary = run_pipeline(
                        "txt_file",
                        temp_path
                    )

                # Remove temporary file
                os.unlink(temp_path)

                st.success("Analysis completed!")

                st.header("📈 Summary")
                show_summary_metrics(summary, df)

                st.header("📋 Milestone 1 Detailed Results")
                st.dataframe(df, use_container_width=True)

                emotion_results = []
                if run_emotion:
                    valid_rows = [
                        (r["id"], r["input_text"])
                        for _, r in df.iterrows() if r["is_valid"]
                    ][: int(max_emotion_rows)]
                    total_valid = int(df["is_valid"].sum())
                    if total_valid > len(valid_rows):
                        st.info(
                            f"Emotion analysis limited to the first {len(valid_rows)} of "
                            f"{total_valid} valid rows. Raise the cap in the sidebar for more."
                        )
                    with st.spinner("Running BERT + DistilBERT emotion analysis..."):
                        emotion_results = run_emotion_stage(
                            valid_rows, bert_model, bert_tok, distil_model, distil_tok
                        )
                    render_emotion_results(emotion_results, load_errors)

                    if emotion_results:
                        combined = build_combined_report(df, emotion_results)
                        st.download_button(
                            "⬇️ Download full report (CSV)",
                            combined.to_csv(index=False),
                            file_name="mood_mentor_report.csv",
                            mime="text/csv",
                        )

            except Exception as e:

                st.error(
                    f"Unable to process the file: {e}"
                )


# ---------------------------------------------------------
# CSV FILE
# ---------------------------------------------------------

elif mode == "📊 CSV File":

    st.header("📊 Upload CSV Mood Data")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            preview_df = pd.read_csv(uploaded_file)

            st.write("### File Preview")

            st.dataframe(
                preview_df.head(),
                use_container_width=True
            )

            st.write(
                f"Rows: {len(preview_df)}"
            )

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )

        if st.button("🔍 Analyze CSV", type="primary"):

            try:

                # Reset file position
                uploaded_file.seek(0)

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".csv"
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                with st.spinner("Analyzing CSV..."):

                    df, summary = run_pipeline(
                        "csv_file",
                        temp_path
                    )

                os.unlink(temp_path)

                st.success("CSV analysis completed!")

                st.header("📈 Summary")
                show_summary_metrics(summary, df)

                st.header("📋 Milestone 1 Detailed Results")
                st.dataframe(df, use_container_width=True)

                emotion_results = []
                if run_emotion:
                    valid_rows = [
                        (r["id"], r["input_text"])
                        for _, r in df.iterrows() if r["is_valid"]
                    ][: int(max_emotion_rows)]
                    total_valid = int(df["is_valid"].sum())
                    if total_valid > len(valid_rows):
                        st.info(
                            f"Emotion analysis limited to the first {len(valid_rows)} of "
                            f"{total_valid} valid rows. Raise the cap in the sidebar for more."
                        )
                    with st.spinner("Running BERT + DistilBERT emotion analysis..."):
                        emotion_results = run_emotion_stage(
                            valid_rows, bert_model, bert_tok, distil_model, distil_tok
                        )
                    render_emotion_results(emotion_results, load_errors)

                    if emotion_results:
                        combined = build_combined_report(df, emotion_results)
                        st.download_button(
                            "⬇️ Download full report (CSV)",
                            combined.to_csv(index=False),
                            file_name="mood_mentor_report.csv",
                            mime="text/csv",
                        )

            except Exception as e:

                st.error(
                    f"Unable to analyze CSV: {e}"
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Mood Mentor — Milestone 1 (Ingestion + Preprocessing + VADER) "
    "+ Milestone 2 (BERT / DistilBERT Emotion Classification + Confidence Scores)"
)