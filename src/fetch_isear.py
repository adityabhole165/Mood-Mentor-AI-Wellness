"""
fetch_isear.py
Downloads an ISEAR dataset from Hugging Face and creates a held-out subset
with the columns expected by src.isear_validation.py.

Run:
    python -m src.fetch_isear --size 200 --seed 42

The generated subset is NOT used for model training, so it remains a separate
benchmark/evaluation corpus.
"""
from __future__ import annotations

import argparse
import os
import pandas as pd


def fetch_isear_subset(output_path: str, size: int = 200, seed: int = 42) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package from requirements.txt first.") from exc

    ds = load_dataset("gsri-18/ISEAR-dataset-complete", split="train")
    df = ds.to_pandas()

    # The public mirror uses emotion/content; support the original-style SIT/EMOT
    # shape as well so this utility remains portable across ISEAR mirrors.
    if {"emotion", "content"}.issubset(df.columns):
        df = df.rename(columns={"emotion": "label", "content": "text"})
    elif {"SIT", "EMOT"}.issubset(df.columns):
        mapping = {
            1: "joy", 2: "fear", 3: "anger", 4: "sadness",
            5: "disgust", 6: "shame", 7: "guilt",
        }
        df = df.rename(columns={"SIT": "text", "EMOT": "label"})
        df["label"] = df["label"].map(mapping)
    else:
        raise ValueError(f"Unsupported ISEAR columns: {list(df.columns)}")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[(df["text"] != "") & df["label"].isin(
        ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
    )]

    # Stratified sample keeps all seven ISEAR labels represented where possible.
    n = min(size, len(df))
    sampled_parts = []
    per_label = max(1, n // df["label"].nunique())
    for label, group in df.groupby("label"):
        sampled_parts.append(group.sample(min(per_label, len(group)), random_state=seed))
    subset = pd.concat(sampled_parts).drop_duplicates("text")
    if len(subset) < n:
        remainder = df[~df["text"].isin(subset["text"])]
        subset = pd.concat([subset, remainder.sample(min(n-len(subset), len(remainder)), random_state=seed)])
    subset = subset.sample(min(n, len(subset)), random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subset.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "isear_subset.csv")
    print(f"Created {fetch_isear_subset(path, args.size, args.seed)}")


if __name__ == "__main__":
    main()
