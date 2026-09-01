"""
ingestion.py
Milestone 1 - Task 1: Validate Text Ingestion Workflow

Responsible for every way text can enter the pipeline:
    1. Direct text entry        (e.g. typed into a chat box)
    2. .txt file upload
    3. .csv file upload         (one text column + optional metadata columns)

Every ingestion path funnels into the same normalized record shape so that
downstream modules (preprocessing, sentiment, reporting) never need to know
where the text came from:

    {
        "id": int,
        "source": "raw_text" | "txt_file" | "csv_file",
        "raw_text": str,
        "char_count": int,
        "is_valid": bool,
        "error": str | None
    }
"""

from __future__ import annotations
import os
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class IngestedRecord:
    id: int
    source: str
    raw_text: str
    char_count: int
    is_valid: bool
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


MIN_CHARS = 1
MAX_CHARS = 50_000  # guard against pathological single inputs


def validate_text(text) -> tuple[bool, Optional[str]]:
    """
    Returns (is_valid, error_message).
    Single source of truth for what counts as usable input -- every
    ingestion path (raw / txt / csv) calls this.
    """
    if text is None:
        return False, "Input is None"

    if not isinstance(text, str):
        return False, f"Input is not a string (got {type(text).__name__})"

    stripped = text.strip()

    if len(stripped) == 0:
        return False, "Input is empty or whitespace-only"

    if len(stripped) < MIN_CHARS:
        return False, "Input shorter than minimum allowed length"

    if len(stripped) > MAX_CHARS:
        return False, f"Input exceeds max length of {MAX_CHARS} characters"

    # Reject inputs that are purely punctuation/symbols with no letters,
    # digits, or emoji-range characters at all (genuinely non-textual noise)
    has_content = any(ch.isalnum() or ord(ch) > 0x2600 for ch in stripped)
    if not has_content:
        return False, "Input contains no analyzable text (symbols/punctuation only)"

    return True, None


def ingest_raw_text(text: str, start_id: int = 0) -> IngestedRecord:
    """Path 1: user types/pastes text directly (e.g. chat input)."""
    is_valid, error = validate_text(text)
    safe_text = text if isinstance(text, str) else ""
    return IngestedRecord(
        id=start_id, source="raw_text", raw_text=safe_text,
        char_count=len(safe_text), is_valid=is_valid, error=error,
    )


def ingest_txt_file(path: str, start_id: int = 0, split_by: str = "line") -> List[IngestedRecord]:
    """
    Path 2: .txt file upload.
    split_by="line"  -> each non-blank line becomes its own record (chat-log style)
    split_by="whole" -> entire file content becomes one record (essay/journal style)
    """
    if not os.path.exists(path):
        return [IngestedRecord(start_id, "txt_file", "", 0, False, f"File not found: {path}")]

    if not path.lower().endswith(".txt"):
        return [IngestedRecord(start_id, "txt_file", "", 0, False, "File is not a .txt file")]

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return [IngestedRecord(start_id, "txt_file", "", 0, False, f"Could not read file: {e}")]

    records = []
    if split_by == "whole":
        is_valid, error = validate_text(content)
        records.append(IngestedRecord(start_id, "txt_file", content, len(content), is_valid, error))
    else:
        lines = content.splitlines()
        if not lines:
            records.append(IngestedRecord(start_id, "txt_file", "", 0, False, "File is empty"))
        for i, line in enumerate(lines):
            is_valid, error = validate_text(line)
            records.append(IngestedRecord(start_id + i, "txt_file", line, len(line), is_valid, error))

    return records


def ingest_csv_file(path: str, text_column: str = "text", start_id: int = 0) -> List[IngestedRecord]:
    """
    Path 3: .csv file upload. Expects at least one column with raw text in it
    (default column name "text"; override with text_column=... for other schemas).
    """
    if not os.path.exists(path):
        return [IngestedRecord(start_id, "csv_file", "", 0, False, f"File not found: {path}")]

    if not path.lower().endswith(".csv"):
        return [IngestedRecord(start_id, "csv_file", "", 0, False, "File is not a .csv file")]

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return [IngestedRecord(start_id, "csv_file", "", 0, False, "CSV file has no data")]
    except Exception as e:
        return [IngestedRecord(start_id, "csv_file", "", 0, False, f"Could not parse CSV: {e}")]

    if text_column not in df.columns:
        return [IngestedRecord(
            start_id, "csv_file", "", 0, False,
            f"Column '{text_column}' not found. Available columns: {list(df.columns)}"
        )]

    records = []
    for i, value in enumerate(df[text_column].tolist()):
        text = "" if pd.isna(value) else str(value)
        is_valid, error = validate_text(text)
        records.append(IngestedRecord(start_id + i, "csv_file", text, len(text), is_valid, error))

    return records


def read_input_data(source_type: str, payload) -> List[IngestedRecord]:
    """
    Single entry point the rest of the app calls, regardless of which of the
    3 supported input methods is being used.
    source_type: "raw_text" | "txt_file" | "csv_file"
    payload:     str (raw text) or file path (for the file-based sources)
    """
    if source_type == "raw_text":
        return [ingest_raw_text(payload)]
    elif source_type == "txt_file":
        return ingest_txt_file(payload)
    elif source_type == "csv_file":
        return ingest_csv_file(payload)
    else:
        return [IngestedRecord(0, source_type, "", 0, False, f"Unsupported source_type: {source_type}")]


if __name__ == "__main__":
    print(ingest_raw_text("I feel really good about today!").to_dict())
    print(ingest_raw_text("   ").to_dict())
    print(ingest_raw_text(None).to_dict())
