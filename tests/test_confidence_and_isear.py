"""
test_confidence_and_isear.py
Milestone 2 - supporting unit tests for Task 4 (confidence) and Task 6 (ISEAR)

No transformer model is loaded here -- these test the pure logic (score
validation, primary-emotion selection, ISEAR label exclusion) in isolation,
mirroring how tests/test_sentiment.py tests classify_compound() in isolation
in Milestone 1.
"""

import os
import pytest
from dataclasses import dataclass
from typing import Dict, List

from src.confidence import build_confidence_report, DEFAULT_THRESHOLD
from src.emotion_dataset import EMOTIONS
from src.isear_validation import (
    load_isear_subset, validate_against_isear, OVERLAPPING_LABELS, EXCLUDED_ISEAR_LABELS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---- Task 4: confidence report ----

def test_build_confidence_report_picks_correct_primary():
    scores = {"joy": 0.81, "sadness": 0.04, "anger": 0.02, "fear": 0.63, "surprise": 0.11, "disgust": 0.01}
    report = build_confidence_report(scores)
    assert report.primary_emotion == "joy"
    assert report.primary_confidence == 0.81
    assert set(report.triggered_emotions) == {"joy", "fear"}  # both cross the 0.5 threshold
    assert report.threshold == DEFAULT_THRESHOLD


def test_build_confidence_report_no_triggered_emotions_below_threshold():
    scores = {e: 0.1 for e in EMOTIONS}
    scores["joy"] = 0.3  # highest, but still below 0.5
    report = build_confidence_report(scores)
    assert report.primary_emotion == "joy"
    assert report.triggered_emotions == []


# ---- Task 6: ISEAR loading + label exclusion ----

def test_load_isear_subset_lowercases_labels():
    df = load_isear_subset(os.path.join(DATA_DIR, "isear_subset_sample.csv"))
    assert (df["label"] == df["label"].str.lower()).all()


def test_isear_excludes_shame_and_guilt():
    assert EXCLUDED_ISEAR_LABELS == {"shame", "guilt"}
    assert "shame" not in OVERLAPPING_LABELS
    assert "guilt" not in OVERLAPPING_LABELS


def test_isear_surprise_is_not_validatable():
    assert "surprise" not in OVERLAPPING_LABELS


@dataclass
class _FakePrediction:
    text: str
    scores: Dict[str, float]
    primary_emotion: str
    primary_confidence: float
    triggered_emotions: List[str]
    threshold: float


def test_validate_against_isear_excludes_shame_guilt_rows():
    df = load_isear_subset(os.path.join(DATA_DIR, "isear_subset_sample.csv"))
    n_shame_guilt = len(df[df["label"].isin(EXCLUDED_ISEAR_LABELS)])
    assert n_shame_guilt > 0  # sanity check the sample data actually has some

    def _always_correct_predict_fn(text, model, tokenizer, threshold=0.5):
        row = df[df["text"] == text].iloc[0]
        label = row["label"]
        scores = {e: (0.9 if e == label else 0.05) for e in EMOTIONS}
        return _FakePrediction(text, scores, label, 0.9, [label], threshold)

    result = validate_against_isear(model=None, tokenizer=None, isear_df=df, predict_fn=_always_correct_predict_fn)
    assert result.n_total_rows == len(df)
    assert result.n_excluded_rows == n_shame_guilt
    assert result.n_validated_rows == len(df) - n_shame_guilt
    assert result.overall_accuracy == 1.0  # every non-excluded row was predicted correctly
