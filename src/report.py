"""
report.py
Milestone 1 - Task 4: Initial Emotion/Sentiment Report Validation

Turns a batch of fully-processed records into:
    1. A row-level DataFrame (input text, processed text, sentiment label/scores)
    2. A summary dict (counts, percentages, sample size) usable to sanity-check
       the run before it's signed off as the Milestone 1 baseline.
"""

from __future__ import annotations
import pandas as pd
from typing import List, Dict


REPORT_COLUMNS = [
    "id", "source", "input_text", "processed_text",
    "sentiment_label", "compound", "pos", "neg", "neu",
    "is_valid", "error",
]


def build_report_dataframe(pipeline_records: List[dict]) -> pd.DataFrame:
    """
    pipeline_records: list of dicts, each merging the ingestion record,
    the preprocessing result, and the sentiment result for one input
    (this is exactly what pipeline.py produces per item).
    """
    rows = []
    for rec in pipeline_records:
        rows.append({
            "id": rec.get("id"),
            "source": rec.get("source"),
            "input_text": rec.get("raw_text", ""),
            "processed_text": rec.get("processed_text", ""),
            "sentiment_label": rec.get("sentiment_label"),
            "compound": rec.get("compound"),
            "pos": rec.get("pos"),
            "neg": rec.get("neg"),
            "neu": rec.get("neu"),
            "is_valid": rec.get("is_valid"),
            "error": rec.get("error"),
        })
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def summarize_report(df: pd.DataFrame) -> Dict:
    """Aggregate stats used to validate the Milestone 1 run at a glance."""
    valid_df = df[df["is_valid"] == True]  # noqa: E712
    total = len(df)
    total_valid = len(valid_df)

    label_counts = valid_df["sentiment_label"].value_counts().to_dict()

    return {
        "total_samples": total,
        "valid_samples": total_valid,
        "invalid_samples": total - total_valid,
        "positive_count": label_counts.get("positive", 0),
        "negative_count": label_counts.get("negative", 0),
        "neutral_count": label_counts.get("neutral", 0),
        "avg_compound_score": round(valid_df["compound"].mean(), 4) if total_valid else None,
    }


def generate_report(pipeline_records: List[dict], output_csv_path: str = None):
    df = build_report_dataframe(pipeline_records)
    summary = summarize_report(df)
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
    return df, summary


def print_summary(summary: Dict) -> None:
    print("=" * 50)
    print("MILESTONE 1 - BASELINE SENTIMENT REPORT SUMMARY")
    print("=" * 50)
    print(f"Total samples ingested : {summary['total_samples']}")
    print(f"Valid samples analyzed : {summary['valid_samples']}")
    print(f"Invalid/skipped        : {summary['invalid_samples']}")
    print(f"  Positive : {summary['positive_count']}")
    print(f"  Negative : {summary['negative_count']}")
    print(f"  Neutral  : {summary['neutral_count']}")
    print(f"Average compound score : {summary['avg_compound_score']}")
    print("=" * 50)
