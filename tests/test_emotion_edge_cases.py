"""
test_emotion_edge_cases.py
Milestone 2 - Task 8: Model Testing & Edge Case Validation

Fast tests that don't require downloading/loading a real BERT model -- they
exercise the SAME code paths (confidence validation, pipeline_v2's per-record
handling) using a fake predict_fn, so this suite runs in CI without a GPU or
network access to HuggingFace. Swap _fake_predict_fn for emotion_bert.predict
/ emotion_distilbert.predict to run the identical assertions against a real
fine-tuned model.
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List

from src.confidence import validate_scores_are_dynamic
from src.ingestion import ingest_raw_text
from src.pipeline_v2 import process_single_record_v2
from src.emotion_dataset import EMOTIONS


@dataclass
class _FakePrediction:
    text: str
    scores: Dict[str, float]
    primary_emotion: str
    primary_confidence: float
    triggered_emotions: List[str]
    threshold: float


def _fake_predict_fn(text, model, tokenizer, threshold: float = 0.5):
    """Deterministic-but-not-identical scores, keyed off text length, so it
    behaves like a real model for test purposes without loading one."""
    base = (len(text) % 10) / 10.0
    scores = {e: max(0.01, min(0.99, base + i * 0.07)) for i, e in enumerate(EMOTIONS)}
    primary = max(scores, key=scores.get)
    triggered = [e for e, s in scores.items() if s >= threshold]
    return _FakePrediction(text, scores, primary, scores[primary], triggered, threshold)


# ---- Task 8 input categories ----

@pytest.mark.parametrize("text", [
    "I am so happy and grateful today!",                              # positive
    "This is the worst day of my life, I hate everything.",           # negative
    "The meeting starts at 3pm in room 4.",                            # neutral
    "I'm thrilled about the promotion but terrified of failing.",     # mixed emotions
    "ok",                                                               # short text
    "I woke up this morning feeling a strange mix of hope and dread " * 5,  # long text
    "omg this is lit, i'm literally shook rn lmaooo",                  # informal text
    "I love this so much!! (emoji-heavy in the real UI)",              # emoji-style
    "Well, that happened.",                                             # ambiguous
])
def test_valid_inputs_produce_result(text):
    record = ingest_raw_text(text)
    result = process_single_record_v2(record, emotion_model=None, emotion_tokenizer=None, predict_fn=_fake_predict_fn)
    assert result["is_valid"] is True
    assert result["primary_emotion"] in EMOTIONS
    assert 0.0 <= result["primary_emotion_confidence"] <= 1.0


def test_empty_input_is_handled_not_crashed():
    record = ingest_raw_text("")
    result = process_single_record_v2(record, emotion_model=None, emotion_tokenizer=None, predict_fn=_fake_predict_fn)
    assert result["is_valid"] is False
    assert result["primary_emotion"] is None
    assert result["error"] is not None


def test_invalid_input_type_is_handled_not_crashed():
    record = ingest_raw_text(None)
    result = process_single_record_v2(record, emotion_model=None, emotion_tokenizer=None, predict_fn=_fake_predict_fn)
    assert result["is_valid"] is False
    assert result["primary_emotion"] is None


def test_model_exception_is_caught_and_surfaced_not_crashed():
    def _broken_predict_fn(text, model, tokenizer, threshold=0.5):
        raise RuntimeError("simulated model failure")

    record = ingest_raw_text("some text")
    result = process_single_record_v2(record, emotion_model=None, emotion_tokenizer=None, predict_fn=_broken_predict_fn)
    assert result["is_valid"] is False
    assert "simulated model failure" in result["error"]


# ---- Task 4/8 overlap: confidence scores must never look hardcoded ----

def test_identical_scores_are_rejected_as_likely_hardcoded():
    flat_scores = {e: 0.5 for e in EMOTIONS}
    with pytest.raises(ValueError):
        validate_scores_are_dynamic(flat_scores)


def test_out_of_range_scores_are_rejected():
    bad_scores = {e: 1.5 for e in EMOTIONS}
    bad_scores[EMOTIONS[0]] = -0.2
    with pytest.raises(ValueError):
        validate_scores_are_dynamic(bad_scores)
