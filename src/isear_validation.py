"""
isear_validation.py
Milestone 2 - Task 6: ISEAR Benchmark Validation

ISEAR (International Survey on Emotion Antecedents and Reactions) is a
classic ~7,000-sentence emotion dataset with SINGLE-label annotations across
7 categories: joy, fear, anger, sadness, disgust, shame, guilt.

IMPORTANT MISMATCH TO KNOW ABOUT (don't skip this in your report):
    - ISEAR has NO "surprise" label -> our model's surprise predictions
      simply can't be checked against ISEAR at all.
    - ISEAR HAS "shame" and "guilt" -> our project's 6-category spec doesn't,
      so those rows are excluded from validation (see EXCLUDED_ISEAR_LABELS).
    -> Validation below only covers the 5 overlapping categories:
       joy, sadness, anger, fear, disgust.

Get the real dataset (not bundled here for licensing reasons) from the
ISEAR project page, or a mirrored copy on Kaggle / HuggingFace Datasets
(search "isear"). Save it as data/isear_subset.csv with columns:
    text,label
(label = one of the 7 ISEAR category names, lowercase). A small hand-made
data/isear_subset_sample.csv is included so this module runs out of the box
before you swap in the real benchmark subset.
"""

from __future__ import annotations
import os
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Callable

from .emotion_dataset import EMOTIONS

EXCLUDED_ISEAR_LABELS = {"shame", "guilt"}           # not in our 6-category spec
UNVALIDATABLE_OUR_LABELS = {"surprise"}               # not in ISEAR's 7 categories
OVERLAPPING_LABELS = [e for e in EMOTIONS if e not in UNVALIDATABLE_OUR_LABELS]


@dataclass
class IsearValidationResult:
    n_total_rows: int
    n_excluded_rows: int
    n_validated_rows: int
    overall_accuracy: float
    per_emotion_accuracy: Dict[str, float]
    misclassified_examples: List[Dict]

    def to_dict(self):
        return asdict(self)


def load_isear_subset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("ISEAR subset CSV must have 'text' and 'label' columns")
    df["label"] = df["label"].str.lower().str.strip()
    return df


def validate_against_isear(
    model, tokenizer, isear_df: pd.DataFrame, predict_fn: Callable, threshold: float = 0.5,
) -> IsearValidationResult:
    """
    predict_fn: emotion_bert.predict or emotion_distilbert.predict --
    whichever model Task 6 says to validate ("the selected model").
    """
    n_total = len(isear_df)
    excluded_df = isear_df[
    isear_df["label"].isin(EXCLUDED_ISEAR_LABELS)
    ].copy()

    usable_df = isear_df[
        ~isear_df["label"].isin(EXCLUDED_ISEAR_LABELS)
    ].copy()

    n_excluded = len(excluded_df)

    # Only overlapping labels are actually used for accuracy validation.
    validation_df = usable_df[
        usable_df["label"].isin(OVERLAPPING_LABELS)
    ].copy()

    correct = 0
    per_emotion_correct = {e: 0 for e in OVERLAPPING_LABELS}
    per_emotion_total = {e: 0 for e in OVERLAPPING_LABELS}
    misclassified = []

    for _, row in validation_df.iterrows():
        text, true_label = row["text"], row["label"]
        prediction = predict_fn(text, model, tokenizer, threshold=threshold)

        per_emotion_total[true_label] += 1
        is_correct = prediction.primary_emotion == true_label
        if is_correct:
            correct += 1
            per_emotion_correct[true_label] += 1
        else:
            misclassified.append({
                "text": text, "expected": true_label,
                "predicted": prediction.primary_emotion,
                "confidence": prediction.primary_confidence,
            })

    per_emotion_accuracy = {
        e: (per_emotion_correct[e] / per_emotion_total[e]) if per_emotion_total[e] else None
        for e in OVERLAPPING_LABELS
    }

    return IsearValidationResult(
        n_total_rows=n_total,
        n_excluded_rows=n_excluded,
        n_validated_rows=len(usable_df),
        overall_accuracy=(correct / len(validation_df)) if len(validation_df) else 0.0,        per_emotion_accuracy=per_emotion_accuracy,
        misclassified_examples=misclassified[:10],  # cap for readability
    )


def print_isear_report(result: IsearValidationResult) -> None:
    print("=" * 50)
    print("ISEAR BENCHMARK VALIDATION")
    print("=" * 50)
    print(f"Total rows in subset     : {result.n_total_rows}")
    print(f"Excluded (label mismatch): {result.n_excluded_rows}")
    print(f"Validated rows           : {result.n_validated_rows}")
    print(f"Overall accuracy         : {result.overall_accuracy:.4f}")
    print("Per-emotion accuracy:")
    for e, acc in result.per_emotion_accuracy.items():
        acc_str = "n/a" if acc is None else f"{acc:.4f}"
        print(f"  {e:10s}: {acc_str}")
    if result.misclassified_examples:
        print("\nSample misclassifications:")
        for m in result.misclassified_examples[:5]:
            print(f"  expected={m['expected']:8s} predicted={m['predicted']:8s} | {m['text'][:60]}")
    print("=" * 50)


if __name__ == "__main__":
    from . import emotion_bert

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    real_path = os.path.join(base_dir, "data", "isear_subset.csv")
    sample_path = os.path.join(base_dir, "data", "isear_subset_sample.csv")
    isear_path = real_path if os.path.exists(real_path) else sample_path
    if isear_path == sample_path:
        print("WARNING: using the bundled smoke-test fixture, not the real ISEAR benchmark.")
        print("For Task 6 final validation, place the held-out ISEAR subset at data/isear_subset.csv.")

    df = load_isear_subset(isear_path)
    model, tokenizer = emotion_bert.load_trained_model()
    result = validate_against_isear(model, tokenizer, df, emotion_bert.predict)
    print_isear_report(result)
