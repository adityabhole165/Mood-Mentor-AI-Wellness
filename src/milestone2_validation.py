"""
milestone2_validation.py
Runs the non-model unit tests and, when trained model artifacts exist, runs
BERT/DistilBERT predictions, evaluation, and ISEAR validation.

This is intentionally a validation runner, not a source of hardcoded results.
"""
from __future__ import annotations

import os
import subprocess
import sys


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    base = project_root()
    print("=" * 64)
    print("MOOD MENTOR - MILESTONE 2 VALIDATION")
    print("=" * 64)

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=base,
        check=False,
    )
    if test.returncode != 0:
        print("\nUnit tests failed. Fix those failures before model validation.")
        return test.returncode

    bert_dir = os.path.join(base, "models", "bert_emotion")
    distil_dir = os.path.join(base, "models", "distilbert_emotion")
    if not (os.path.isdir(bert_dir) and os.path.isdir(distil_dir)):
        print("\nAll unit tests passed.")
        print("Model artifacts are not present yet.")
        print("Run `python -m src.train_models` to fine-tune both models, then")
        print("run this validator again for Tasks 5-7 and final model validation.")
        return 0

    from .emotion_dataset import load_emotion_dataset
    from .emotion_bert import load_trained_model as load_bert
    from .emotion_distilbert import load_trained_model as load_distil
    from .evaluation import split_labeled_data, evaluate_both_models, print_evaluation_report
    from .isear_validation import load_isear_subset, validate_against_isear, print_isear_report
    from . import emotion_bert, emotion_distilbert

    data_csv = os.path.join(base, "data", "emotion_train_sample.csv")
    texts, labels = load_emotion_dataset(data_csv)
    _, heldout_texts, _, heldout_labels = split_labeled_data(texts, labels)

    bert_model, bert_tok = load_bert(bert_dir)
    distil_model, distil_tok = load_distil(distil_dir)

    comparison = evaluate_both_models(
        bert_model, bert_tok, distil_model, distil_tok,
        heldout_texts, heldout_labels,
    )
    print("\nMODEL COMPARISON")
    print(comparison)

    isear_path = os.path.join(base, "data", "isear_subset.csv")
    if not os.path.exists(isear_path):
        isear_path = os.path.join(base, "data", "isear_subset_sample.csv")
        print("\nWARNING: ISEAR validation is using the smoke-test fixture.")
    df = load_isear_subset(isear_path)

    # Validate the selected model (the comparison selects by macro F1).
    selected = comparison["better_model"]
    if selected == "bert":
        result = validate_against_isear(bert_model, bert_tok, df, emotion_bert.predict)
    else:
        result = validate_against_isear(distil_model, distil_tok, df, emotion_distilbert.predict)
    print_isear_report(result)
    print("\nMilestone 2 validation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
