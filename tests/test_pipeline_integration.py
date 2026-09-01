"""Task 5 - Complete Pipeline Integration Testing"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import run_pipeline, run_pipeline_multi


def test_raw_text_flows_through_entire_pipeline():
    df, summary = run_pipeline("raw_text", "I am absolutely thrilled about this news!")
    assert len(df) == 1
    assert df.iloc[0]["is_valid"] == True  # noqa: E712
    assert df.iloc[0]["sentiment_label"] == "positive"
    assert df.iloc[0]["processed_text"] != ""

def test_txt_file_flows_through_entire_pipeline(tmp_path):
    p = tmp_path / "journal.txt"
    p.write_text("I feel wonderful today\nI feel terrible today\nThe meeting is at 3pm tomorrow")
    df, summary = run_pipeline("txt_file", str(p))
    assert summary["total_samples"] == 3
    assert summary["valid_samples"] == 3
    assert set(df["sentiment_label"]) == {"positive", "negative", "neutral"}

def test_csv_file_flows_through_entire_pipeline(tmp_path):
    p = tmp_path / "corpus.csv"
    pd.DataFrame({"text": ["I love my job!", "I dread Mondays.", ""]}).to_csv(p, index=False)
    df, summary = run_pipeline("csv_file", str(p))
    assert summary["total_samples"] == 3
    assert summary["invalid_samples"] == 1  # the empty row

def test_invalid_input_does_not_break_the_pipeline():
    df, summary = run_pipeline("raw_text", "")
    assert summary["total_samples"] == 1
    assert summary["valid_samples"] == 0
    assert df.iloc[0]["sentiment_label"] is None

def test_module_output_correctly_feeds_next_module():
    """Checks the actual handoffs: ingestion.raw_text -> preprocessing.processed_text
    -> sentiment.label, all present together on the same row."""
    df, _ = run_pipeline("raw_text", "I am incredibly grateful for everything today!")
    row = df.iloc[0]
    assert row["input_text"] == "I am incredibly grateful for everything today!"
    assert "grateful" in row["processed_text"]
    assert row["sentiment_label"] == "positive"
    assert row["compound"] > 0

def test_multiple_mixed_source_inputs_in_one_run(tmp_path):
    txt_p = tmp_path / "j.txt"
    txt_p.write_text("Feeling great about the new job")
    csv_p = tmp_path / "c.csv"
    pd.DataFrame({"text": ["Nervous about the interview"]}).to_csv(csv_p, index=False)

    df, summary = run_pipeline_multi(sources=[
        ("raw_text", "This week has been fantastic"),
        ("txt_file", str(txt_p)),
        ("csv_file", str(csv_p)),
    ])
    assert summary["total_samples"] == 3
    assert set(df["source"]) == {"raw_text", "txt_file", "csv_file"}

def test_report_csv_output_written_from_full_pipeline(tmp_path):
    out_path = tmp_path / "milestone1.csv"
    df, summary = run_pipeline("raw_text", "Excited for the launch!", output_csv_path=str(out_path))
    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert len(written) == 1

def test_end_to_end_sample_corpus_and_journal_files():
    """Full Milestone 1 smoke test using the checked-in sample data."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df, summary = run_pipeline_multi(sources=[
        ("txt_file", os.path.join(base_dir, "data", "sample_journal.txt")),
        ("csv_file", os.path.join(base_dir, "data", "sample_corpus.csv")),
    ])
    assert summary["total_samples"] == 22  # 10 journal lines + 12 csv rows
    assert summary["valid_samples"] == 20  # 2 deliberately invalid rows in the csv
    assert summary["positive_count"] + summary["negative_count"] + summary["neutral_count"] == 20
