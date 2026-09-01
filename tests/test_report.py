"""Task 4 - Initial Emotion/Sentiment Report Validation"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.report import build_report_dataframe, summarize_report, generate_report, REPORT_COLUMNS


SAMPLE_RECORDS = [
    {"id": 0, "source": "raw_text", "raw_text": "I love this!", "processed_text": "love",
     "sentiment_label": "positive", "compound": 0.6, "pos": 0.7, "neg": 0.0, "neu": 0.3,
     "is_valid": True, "error": None},
    {"id": 1, "source": "raw_text", "raw_text": "I hate this!", "processed_text": "hate",
     "sentiment_label": "negative", "compound": -0.6, "pos": 0.0, "neg": 0.7, "neu": 0.3,
     "is_valid": True, "error": None},
    {"id": 2, "source": "raw_text", "raw_text": "The sky is blue.", "processed_text": "sky blue",
     "sentiment_label": "neutral", "compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0,
     "is_valid": True, "error": None},
    {"id": 3, "source": "raw_text", "raw_text": "", "processed_text": "",
     "sentiment_label": None, "compound": None, "pos": None, "neg": None, "neu": None,
     "is_valid": False, "error": "Input is empty or whitespace-only"},
]


def test_report_dataframe_has_expected_columns():
    df = build_report_dataframe(SAMPLE_RECORDS)
    assert list(df.columns) == REPORT_COLUMNS

def test_report_dataframe_row_count_matches_input():
    df = build_report_dataframe(SAMPLE_RECORDS)
    assert len(df) == len(SAMPLE_RECORDS)

def test_report_preserves_input_and_processed_text():
    df = build_report_dataframe(SAMPLE_RECORDS)
    assert df.iloc[0]["input_text"] == "I love this!"
    assert df.iloc[0]["processed_text"] == "love"

def test_summary_counts_labels_correctly():
    df = build_report_dataframe(SAMPLE_RECORDS)
    summary = summarize_report(df)
    assert summary["positive_count"] == 1
    assert summary["negative_count"] == 1
    assert summary["neutral_count"] == 1
    assert summary["invalid_samples"] == 1
    assert summary["total_samples"] == 4
    assert summary["valid_samples"] == 3

def test_summary_average_compound_computed_not_hardcoded():
    df = build_report_dataframe(SAMPLE_RECORDS)
    summary = summarize_report(df)
    expected_avg = round((0.6 + -0.6 + 0.0) / 3, 4)
    assert summary["avg_compound_score"] == expected_avg

def test_generate_report_writes_csv(tmp_path):
    out_path = tmp_path / "report.csv"
    df, summary = generate_report(SAMPLE_RECORDS, output_csv_path=str(out_path))
    assert out_path.exists()
    assert len(df) == 4

def test_summary_handles_all_invalid_records_without_crash():
    all_invalid = [SAMPLE_RECORDS[3]]
    df = build_report_dataframe(all_invalid)
    summary = summarize_report(df)
    assert summary["valid_samples"] == 0
    assert summary["avg_compound_score"] is None
