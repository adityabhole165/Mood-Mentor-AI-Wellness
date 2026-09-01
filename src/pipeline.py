"""
pipeline.py
Milestone 1 - Task 5: Complete Pipeline Integration

Wires ingestion -> preprocessing -> sentiment -> report into one callable,
and is the module you'd actually import from a FastAPI endpoint or a Spring
Boot service (via REST/Kafka) later.

Each stage's failure is caught and recorded per-record instead of crashing
the whole batch -- one bad row in a 500-row CSV upload shouldn't kill the run.
"""

from __future__ import annotations
from typing import List, Dict

from .ingestion import read_input_data, IngestedRecord
from .preprocessing import preprocess_text
from .sentiment import analyze_sentiment
from .report import generate_report, print_summary


def process_single_record(record: IngestedRecord) -> Dict:
    """Push one ingested record through preprocessing + sentiment."""
    base = record.to_dict()

    if not record.is_valid:
        # Task 1 requirement: invalid input must be handled, not crash downstream stages
        base.update({
            "processed_text": "", "sentiment_label": None,
            "compound": None, "pos": None, "neg": None, "neu": None,
        })
        return base

    try:
        processed = preprocess_text(record.raw_text)
        # sentiment runs on cleaned_text (case/punctuation/emoji preserved) -- see sentiment.py docstring
        sentiment = analyze_sentiment(processed.cleaned_text)

        base.update({
            "processed_text": processed.processed_text,
            "sentiment_label": sentiment.label,
            "compound": sentiment.compound,
            "pos": sentiment.pos, "neg": sentiment.neg, "neu": sentiment.neu,
        })
    except Exception as e:
        # integration failure between modules -- surface it, don't hide it
        base.update({
            "is_valid": False,
            "error": f"Pipeline error at preprocessing/sentiment stage: {e}",
            "processed_text": "", "sentiment_label": None,
            "compound": None, "pos": None, "neg": None, "neu": None,
        })

    return base


def run_pipeline(source_type: str, payload, output_csv_path: str = None):
    """
    source_type: "raw_text" | "txt_file" | "csv_file"
    payload:     text string, or a file path
    Returns (dataframe, summary_dict) -- ready for Task 4 report validation.
    """
    ingested_records = read_input_data(source_type, payload)
    processed_records = [process_single_record(r) for r in ingested_records]
    return generate_report(processed_records, output_csv_path)


def run_pipeline_multi(sources: List[tuple], output_csv_path: str = None):
    """
    Run several inputs of possibly different source types through the same
    pipeline and merge into a single report -- this is what Task 5's
    "test with multiple sample inputs" actually validates.
    sources: list of (source_type, payload) tuples
    """
    all_records = []
    next_id = 0
    for source_type, payload in sources:
        recs = read_input_data(source_type, payload)
        for r in recs:
            r.id = next_id
            next_id += 1
        all_records.extend(recs)

    processed_records = [process_single_record(r) for r in all_records]
    return generate_report(processed_records, output_csv_path)


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt_path = os.path.join(base_dir, "data", "sample_journal.txt")
    csv_path = os.path.join(base_dir, "data", "sample_corpus.csv")

    df, summary = run_pipeline_multi(
        sources=[
            ("raw_text", "I'm feeling really good about this project today!"),
            ("raw_text", ""),  # deliberately invalid -- proves Task 1's "handle empty input"
            ("txt_file", txt_path),
            ("csv_file", csv_path),
        ],
        output_csv_path=os.path.join(base_dir, "data", "milestone1_report.csv"),
    )

    print_summary(summary)
    print("\nSample rows:\n")
    print(df.head(10).to_string(index=False))
