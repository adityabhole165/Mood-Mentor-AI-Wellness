"""
evaluation.py
Milestone 2 - Task 5: Model Evaluation & Validation

Computes accuracy, precision, recall, and macro F1-score for a fine-tuned
emotion model against a labeled test set, using scikit-learn so every number
is computed from real predictions vs. real labels -- nothing here is a
made-up/estimated metric.

Multi-label note: "accuracy" for multi-label problems is usually reported as
subset accuracy (exact match of the full 6-label set) AND per-label accuracy
averaged across all labels -- both are returned here so you can report
whichever your milestone write-up wants.
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, asdict
from typing import Dict, List

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class EvaluationResult:
    model_name: str
    accuracy_subset: float      # exact match across all 6 labels
    accuracy_per_label_avg: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    n_samples: int

    def to_dict(self):
        return asdict(self)


def get_predictions(model, tokenizer, texts: List[str], threshold: float = 0.5, max_length: int = 128) -> np.ndarray:
    """Runs the model over every text and returns an (n_samples, 6) binary matrix."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).numpy()
            all_preds.append((probs >= threshold).astype(int))
    return np.array(all_preds)


def evaluate_model(
    model, tokenizer, texts: List[str], true_labels: List[List[float]],
    model_name: str = "model", threshold: float = 0.5,
) -> EvaluationResult:
    """Task 5 core: run the model on a labeled test set and score it."""
    y_true = np.array(true_labels).astype(int)
    y_pred = get_predictions(model, tokenizer, texts, threshold=threshold)

    return EvaluationResult(
        model_name=model_name,
        accuracy_subset=float(accuracy_score(y_true, y_pred)),
        accuracy_per_label_avg=float(np.mean(y_true == y_pred)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        n_samples=len(texts),
    )



def split_labeled_data(texts: List[str], labels: List[List[float]], test_size: float = 0.2, seed: int = 42):
    """Create a deterministic held-out split for honest Milestone 2 evaluation."""
    from sklearn.model_selection import train_test_split
    return train_test_split(texts, labels, test_size=test_size, random_state=seed)


def evaluate_both_models(
    bert_model, bert_tokenizer, distilbert_model, distilbert_tokenizer,
    texts: List[str], labels: List[List[float]], threshold: float = 0.5,
) -> Dict:
    """Evaluate BERT and DistilBERT on the exact same held-out examples."""
    bert = evaluate_model(bert_model, bert_tokenizer, texts, labels, "bert", threshold)
    distil = evaluate_model(distilbert_model, distilbert_tokenizer, texts, labels, "distilbert", threshold)
    return compare_models(bert, distil)

def compare_models(bert_result: EvaluationResult, distilbert_result: EvaluationResult) -> Dict:
    """Task 5: 'Compare both models and identify the better-performing model.'"""
    better = "bert" if bert_result.f1_macro >= distilbert_result.f1_macro else "distilbert"
    return {
        "bert": bert_result.to_dict(),
        "distilbert": distilbert_result.to_dict(),
        "better_model": better,
        "f1_macro_gap": abs(bert_result.f1_macro - distilbert_result.f1_macro),
    }


def print_evaluation_report(result: EvaluationResult) -> None:
    print("=" * 50)
    print(f"EVALUATION -- {result.model_name}")
    print("=" * 50)
    print(f"Samples evaluated       : {result.n_samples}")
    print(f"Subset accuracy (exact) : {result.accuracy_subset:.4f}")
    print(f"Per-label accuracy avg  : {result.accuracy_per_label_avg:.4f}")
    print(f"Precision (macro)       : {result.precision_macro:.4f}")
    print(f"Recall (macro)          : {result.recall_macro:.4f}")
    print(f"F1-score (macro)        : {result.f1_macro:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    import os
    from .emotion_dataset import load_emotion_dataset
    from . import emotion_bert, emotion_distilbert

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(base_dir, "data", "emotion_train_sample.csv")
    texts, labels = load_emotion_dataset(test_csv)
    _, heldout_texts, _, heldout_labels = split_labeled_data(texts, labels)
    # The small sample corpus is only a smoke-test dataset; for a publishable
    # evaluation, replace it with a larger held-out labeled benchmark.
    bert_model, bert_tok = emotion_bert.load_trained_model()
    distil_model, distil_tok = emotion_distilbert.load_trained_model()
    report = evaluate_both_models(
        bert_model, bert_tok, distil_model, distil_tok,
        heldout_texts, heldout_labels,
    )
    print(report)
