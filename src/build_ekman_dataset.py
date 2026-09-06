"""
build_ekman_dataset.py
Builds a real, properly-sized multi-label training CSV for
src/emotion_bert.py / src/emotion_distilbert.py, replacing the ~45-row
hand-made data/emotion_train_sample.csv.

Source: "Jsevisal/go_emotions_ekman" on Hugging Face -- GoEmotions already
re-mapped onto Ekman's basic-emotion taxonomy. Column and class names are
read from ds.features at runtime rather than hardcoded, since the schema
doesn't match what's documented on the dataset card.

Run:
    python -m src.build_ekman_dataset --per_label 500 --seed 42
"""
from __future__ import annotations
import argparse
import os
import pandas as pd

from .emotion_dataset import EMOTIONS  # ["joy","sadness","anger","fear","surprise","disgust"]


def _get_text_column(ds) -> str:
    for candidate in ("text", "sentence", "comment_text"):
        if candidate in ds.features:
            return candidate
    raise ValueError(f"Could not find a text column in {list(ds.features.keys())}")


def _get_label_column(ds):
    """Returns (column_name, [class names in index order]) read from the
    dataset's own schema instead of assumed."""
    for candidate in ("labels_ekman", "labels", "label", "ekman"):
        if candidate in ds.features:
            feature = ds.features[candidate]
            inner = getattr(feature, "feature", feature)  # unwrap Sequence(ClassLabel)
            names = getattr(inner, "names", None)
            if names:
                return candidate, [str(n).lower() for n in names]
    raise ValueError(
        f"Could not find a ClassLabel/Sequence(ClassLabel) label column in "
        f"{list(ds.features.keys())} -- inspect ds.features manually."
    )


def build(output_path: str, per_label: int = 500, seed: int = 42) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package from requirements.txt first.") from exc

    ds = load_dataset("Jsevisal/go_emotions_ekman", split="train")
    text_col = _get_text_column(ds)
    label_col, source_labels = _get_label_column(ds)
    print(f"Detected text column='{text_col}', label column='{label_col}', classes={source_labels}")

    df = ds.to_pandas().rename(columns={text_col: "text"})
    for i, name in enumerate(source_labels):
        df[name] = df[label_col].apply(lambda idxs, i=i: int(i in list(idxs)))

    if "neutral" in df.columns:
        df = df[df["neutral"] == 0].copy()  # drop neutral-only rows -- not one of our 6

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    missing = [e for e in EMOTIONS if e not in df.columns]
    if missing:
        raise ValueError(
            f"Ekman columns missing after extraction: {missing}. "
            f"Detected source classes were {source_labels} -- names must match "
            f"{EMOTIONS} exactly (case-insensitive)."
        )

    sampled_idx = set()
    for emotion in EMOTIONS:
        positive = df[df[emotion] == 1]
        take = positive.sample(min(per_label, len(positive)), random_state=seed)
        sampled_idx.update(take.index.tolist())

    subset = df.loc[sorted(sampled_idx)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    subset = subset[["text"] + EMOTIONS]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subset.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_label", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "emotion_train_sample.csv")
    result = build(path, args.per_label, args.seed)

    from .emotion_dataset import load_emotion_dataset, label_distribution
    texts, labels = load_emotion_dataset(result)
    print(f"Wrote {result}")
    print(f"Total rows: {len(texts)}")
    print("Per-label positive counts:", label_distribution(labels))


if __name__ == "__main__":
    main()