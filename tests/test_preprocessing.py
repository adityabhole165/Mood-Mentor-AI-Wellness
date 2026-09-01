"""Task 2 - Preprocessing Validation"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text, tokenize, remove_stopwords, lemmatize, preprocess_text


def test_tokenization_basic():
    tokens = tokenize("I am happy today")
    assert tokens == ["i", "am", "happy", "today"]

def test_stopword_removal():
    tokens = ["i", "am", "very", "happy", "today"]
    filtered = remove_stopwords(tokens)
    assert "i" not in filtered
    assert "am" not in filtered
    assert "happy" in filtered
    assert "today" in filtered

def test_lemmatization_collapses_verb_forms():
    # "running"/"runs" should both collapse to "run" once POS-tagged correctly
    tokens = ["running", "runs", "runner"]
    lemmas = lemmatize(tokens)
    assert lemmas[0] == "run"
    assert lemmas[1] == "run"

def test_noise_filtering_repeated_spaces():
    cleaned = clean_text("too      many     spaces")
    assert cleaned == "too many spaces"

def test_noise_filtering_control_characters():
    cleaned = clean_text("hello\x00\x1fworld")
    assert "\x00" not in cleaned
    assert "\x1f" not in cleaned

def test_special_characters_and_punctuation_only_filtered_from_stopword_stage():
    result = preprocess_text("!!!???...")
    assert result.processed_text == ""
    assert result.is_empty_after_processing is True

def test_punctuation_preserved_in_cleaned_text_for_vader():
    # cleaned_text must KEEP punctuation -- VADER depends on it
    result = preprocess_text("This is amazing!!!")
    assert "!!!" in result.cleaned_text

def test_empty_text_handled_without_crash():
    result = preprocess_text("")
    assert result.tokens == []
    assert result.processed_text == ""
    assert result.is_empty_after_processing is True

def test_none_input_handled_without_crash():
    result = preprocess_text(None)
    assert result.original_text == ""
    assert result.is_empty_after_processing is True

def test_repeated_spaces_collapsed_end_to_end():
    result = preprocess_text("I     am      so   happy")
    assert "  " not in result.cleaned_text

def test_different_text_lengths():
    short = preprocess_text("Good.")
    long_text = preprocess_text(
        "This has been an incredibly long and emotionally exhausting week "
        "filled with deadlines, meetings, and far too little sleep, and I "
        "am honestly not sure how I am going to make it through the rest of it."
    )
    assert len(short.tokens) < len(long_text.tokens)
    assert short.is_empty_after_processing is False
    assert long_text.is_empty_after_processing is False

def test_processed_output_derived_from_original_not_hardcoded():
    r1 = preprocess_text("I love sunny mornings")
    r2 = preprocess_text("I hate rainy mornings")
    assert r1.processed_text != r2.processed_text
