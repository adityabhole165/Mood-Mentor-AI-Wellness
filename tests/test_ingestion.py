"""Task 1 - Validate Text Ingestion Workflow"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion import ingest_raw_text, ingest_txt_file, ingest_csv_file, read_input_data, validate_text


# ---- Path 1: raw text entry ----

def test_raw_text_valid():
    r = ingest_raw_text("I feel great today!")
    assert r.is_valid is True
    assert r.error is None
    assert r.source == "raw_text"

def test_raw_text_empty_string():
    r = ingest_raw_text("")
    assert r.is_valid is False
    assert "empty" in r.error.lower()

def test_raw_text_whitespace_only():
    r = ingest_raw_text("     ")
    assert r.is_valid is False

def test_raw_text_none_input():
    r = ingest_raw_text(None)
    assert r.is_valid is False

def test_raw_text_non_string_input():
    is_valid, error = validate_text(12345)
    assert is_valid is False
    assert "not a string" in error.lower()

def test_raw_text_punctuation_only_rejected():
    is_valid, _ = validate_text("!!!???...")
    assert is_valid is False

def test_raw_text_emoji_only_accepted():
    # emoji alone still carries emotion signal -> should be analyzable
    is_valid, _ = validate_text("\U0001F604\U0001F604")
    assert is_valid is True


# ---- Path 2: .txt file upload ----

def test_txt_file_line_split(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Line one is happy\nLine two is sad\n\nLine four here")
    records = ingest_txt_file(str(p))
    valid_texts = [r.raw_text for r in records if r.is_valid]
    assert "Line one is happy" in valid_texts
    assert "Line two is sad" in valid_texts

def test_txt_file_whole_mode(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("A whole journal entry across one blob of text.")
    records = ingest_txt_file(str(p), split_by="whole")
    assert len(records) == 1
    assert records[0].is_valid is True

def test_txt_file_missing_returns_error():
    records = ingest_txt_file("/tmp/does_not_exist_12345.txt")
    assert records[0].is_valid is False
    assert "not found" in records[0].error.lower()

def test_txt_file_wrong_extension_rejected(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text("some text")
    records = ingest_txt_file(str(p))
    assert records[0].is_valid is False

def test_txt_file_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    records = ingest_txt_file(str(p))
    assert records[0].is_valid is False


# ---- Path 3: .csv file upload ----

def test_csv_file_valid_rows(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"text": ["I am happy", "I am sad", ""]}).to_csv(p, index=False)
    records = ingest_csv_file(str(p))
    assert len(records) == 3
    assert records[0].is_valid is True
    assert records[1].is_valid is True
    assert records[2].is_valid is False  # empty row correctly flagged

def test_csv_file_missing_column(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"other_col": ["a", "b"]}).to_csv(p, index=False)
    records = ingest_csv_file(str(p))
    assert records[0].is_valid is False
    assert "not found" in records[0].error.lower()

def test_csv_file_missing_path():
    records = ingest_csv_file("/tmp/does_not_exist_12345.csv")
    assert records[0].is_valid is False

def test_csv_file_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    records = ingest_csv_file(str(p))
    assert records[0].is_valid is False

def test_csv_file_custom_column_name(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"message": ["hello world"]}).to_csv(p, index=False)
    records = ingest_csv_file(str(p), text_column="message")
    assert records[0].is_valid is True


# ---- Unified dispatcher ----

def test_read_input_data_dispatches_raw_text():
    records = read_input_data("raw_text", "Some valid input text")
    assert records[0].is_valid is True

def test_read_input_data_unsupported_source():
    records = read_input_data("pdf_file", "irrelevant")
    assert records[0].is_valid is False
    assert "unsupported" in records[0].error.lower()
