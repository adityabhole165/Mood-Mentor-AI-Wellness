"""
pipeline_v2.py
Milestone 2 - Task 7: Milestone 1 & Milestone 2 Integration

Extends src/pipeline.py's run_pipeline() by adding the transformer emotion
stage after VADER, WITHOUT changing Milestone 1's behavior -- this file only
ADDS a stage; it never modifies ingestion.py / preprocessing.py /
sentiment.py / report.py.

Flow:
    Enter Text -> Text Preprocessing -> VADER Sentiment Analysis
    -> BERT/DistilBERT Emotion Analysis -> Emotion Classification
    -> Confidence Scores -> Final Analysis Result
"""

from __future__ import annotations
from typing import Dict, Callable

from .ingestion import read_input_data, IngestedRecord
from .preprocessing import preprocess_text
from .sentiment import analyze_sentiment
from .confidence import build_confidence_report
from . import emotion_bert


def process_single_record_v2(
    record: IngestedRecord,
    emotion_model,
    emotion_tokenizer,
    predict_fn: Callable = emotion_bert.predict,
) -> Dict:
    """Milestone 1's process_single_record(), plus the Milestone 2 emotion stage."""
    base = record.to_dict()

    if not record.is_valid:
        base.update({
            "processed_text": "", "sentiment_label": None,
            "compound": None, "pos": None, "neg": None, "neu": None,
            "primary_emotion": None, "primary_emotion_confidence": None,
            "emotion_scores": None, "triggered_emotions": None,
        })
        return base

    try:
        processed = preprocess_text(record.raw_text)
        sentiment = analyze_sentiment(processed.cleaned_text)

        # Milestone 2 stage -- runs on cleaned_text too, same reasoning as
        # VADER: a transformer tokenizer handles its own subword splitting,
        # but punctuation/case still carries emotional signal ("SO SCARED"
        # vs "so scared"), so we don't feed it the stopword-stripped version.
        emotion_prediction = predict_fn(processed.cleaned_text, emotion_model, emotion_tokenizer)
        confidence = build_confidence_report(emotion_prediction.scores)

        base.update({
            "processed_text": processed.processed_text,
            "sentiment_label": sentiment.label,
            "compound": sentiment.compound,
            "pos": sentiment.pos, "neg": sentiment.neg, "neu": sentiment.neu,
            "primary_emotion": confidence.primary_emotion,
            "primary_emotion_confidence": confidence.primary_confidence,
            "emotion_scores": confidence.all_scores,
            "triggered_emotions": confidence.triggered_emotions,
        })
    except Exception as e:
        # integration failure between modules -- surface it, don't hide it
        base.update({
            "is_valid": False,
            "error": f"Pipeline v2 error at preprocessing/sentiment/emotion stage: {e}",
            "processed_text": "", "sentiment_label": None,
            "compound": None, "pos": None, "neg": None, "neu": None,
            "primary_emotion": None, "primary_emotion_confidence": None,
            "emotion_scores": None, "triggered_emotions": None,
        })

    return base


def run_pipeline_v2(
    source_type: str,
    payload,
    emotion_model_dir: str = emotion_bert.DEFAULT_SAVE_DIR,
    predict_fn: Callable = emotion_bert.predict,
):
    """Milestone 1 + 2, end to end, no manual steps between stages (Task 7 requirement)."""
    emotion_model, emotion_tokenizer = emotion_bert.load_trained_model(emotion_model_dir)

    ingested_records = read_input_data(source_type, payload)
    return [
        process_single_record_v2(r, emotion_model, emotion_tokenizer, predict_fn)
        for r in ingested_records
    ]
    # Feed the returned list of dicts into report.py's build_report_dataframe()
    # if you want the same DataFrame + summary shape Milestone 1's UI expects.


if __name__ == "__main__":
    results = run_pipeline_v2(
        "raw_text",
        "I am excited about the new opportunity but nervous about the outcome.",
    )
    for r in results:
        print(r)
