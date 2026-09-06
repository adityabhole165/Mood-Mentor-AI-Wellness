"""
emotion_dataset.py
Milestone 2 - shared dataset utilities for Task 1 (BERT) and Task 2 (DistilBERT)

Both models fine-tune on the exact same data shape, so this module is the
single source of truth for:
    - the 6 project-specified emotion categories (Task 3)
    - CSV -> (texts, multi-hot labels) loading
    - the torch Dataset wrapper used by Trainer

CSV schema expected (data/emotion_train_sample.csv):
    text, joy, sadness, anger, fear, surprise, disgust
    "I got the job!!",1,0,0,0,0,0
    "I'm thrilled but a bit scared about moving cities",1,0,0,1,0,0

Each emotion column is 0/1 (multi-hot) -- a row can have more than one 1,
which is how "multiple emotions in one sentence" (Task 3) is represented.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, List, Tuple

# torch/Dataset are only needed by EmotionDataset below, imported lazily so
# that modules which only need the EMOTIONS list (confidence.py, tests) can
# be imported without torch installed -- keeps Task 8's fast test suite fast.
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

# The project specifies exactly these six emotion categories (Task 3).
EMOTIONS: List[str] = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]


def load_emotion_dataset(csv_path: str) -> Tuple[List[str], List[List[float]]]:
    """Read the training CSV and return (texts, multi-hot label rows)."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Training dataset not found: {csv_path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Training dataset is empty: {csv_path}") from exc

    missing = [c for c in EMOTIONS if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing emotion columns: {missing}")
    if "text" not in df.columns:
        raise ValueError("Training CSV must have a 'text' column")

    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    for emotion in EMOTIONS:
        if not df[emotion].isin([0, 1]).all():
            raise ValueError(f"Emotion column '{emotion}' must contain only 0/1 labels")
    texts = df["text"].tolist()
    labels = df[EMOTIONS].astype(float).values.tolist()
    return texts, labels


def label_distribution(labels: List[List[float]]) -> Dict[str, int]:
    """Positive-example count per emotion, computed straight from the loaded
    labels -- lets train_models.py print real class balance instead of
    guessing, and catches a too-small/imbalanced CSV before you waste a
    training run on it."""
    import numpy as np
    arr = np.array(labels)
    return {emotion: int(arr[:, i].sum()) for i, emotion in enumerate(EMOTIONS)}


def compute_pos_weight(labels: List[List[float]]):
    """BCEWithLogitsLoss pos_weight = n_negative / n_positive, per label,
    computed dynamically from the actual training labels (never hardcoded).
    This is what stops the model from just learning the majority label and
    collapsing onto it for every input."""
    if torch is None:
        raise ImportError("torch is required to compute pos_weight")
    arr = torch.tensor(labels, dtype=torch.float)
    pos = arr.sum(dim=0).clamp(min=1.0)  # avoid divide-by-zero
    neg = arr.shape[0] - pos
    return neg / pos


class EmotionDataset(Dataset):
    """torch Dataset: tokenizes lazily so it works the same for BERT/DistilBERT."""

    def __init__(self, texts: List[str], labels: List[List[float]], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


if __name__ == "__main__":
    print(f"Emotion categories ({len(EMOTIONS)}): {EMOTIONS}")