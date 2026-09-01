"""
preprocessing.py
Milestone 1 - Task 2: Preprocessing Validation

IMPORTANT DESIGN NOTE (matters for Task 3 too):
VADER's accuracy comes FROM punctuation, capitalization, and emojis
("GREAT!!!" scores higher than "great"). So this module produces TWO
outputs, not one:

    - cleaned_text   -> noise removed only (extra whitespace, control chars).
                        Case, punctuation, and emojis are PRESERVED.
                        -> feed this to VADER in sentiment.py
    - processed_text -> tokenized, lowercased, stopwords removed, lemmatized.
                        -> feed this to a bag-of-words model, a transformer
                           tokenizer, or use it for keyword extraction later

Stripping punctuation/case before VADER is one of the most common bugs in
sentiment pipelines and it quietly tanks accuracy, so don't merge these two
into one "processed" field.
"""

from __future__ import annotations
import re
import string
from dataclasses import dataclass, asdict
from typing import List

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))

# Emoji ranges we deliberately KEEP (they carry emotion signal)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


@dataclass
class ProcessedText:
    original_text: str
    cleaned_text: str          # noise removed, case/punctuation/emoji preserved -> use for VADER
    tokens: List[str]          # word-tokenized, lowercased
    tokens_no_stopwords: List[str]
    lemmatized_tokens: List[str]
    processed_text: str        # lemmatized_tokens joined -> use for bag-of-words / classical ML
    is_empty_after_processing: bool

    def to_dict(self):
        return asdict(self)


def clean_text(text: str) -> str:
    """Remove genuine noise while preserving signal VADER needs."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\x00-\x1f\x7f\u200b\u200c\u200d\ufeff]", " ", text)  # control/invisible chars
    text = re.sub(r"\s+", " ", text)  # collapse repeated whitespace/newlines/tabs
    return text.strip()


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in word_tokenize(text) if tok.strip()]


def _is_punctuation_only(token: str) -> bool:
    """
    Catches BOTH single punctuation chars ('!') AND punctuation runs ('...', '???', em-dash)
    that string.punctuation membership alone misses, while leaving emoji tokens alone.
    """
    if _EMOJI_PATTERN.fullmatch(token):
        return False
    return all(ch in string.punctuation or ch in "\u2014\u2013\u2026\u201c\u201d\u2018\u2019" for ch in token)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOPWORDS and not _is_punctuation_only(t)]


def _wordnet_pos(treebank_tag: str) -> str:
    """Map a Penn Treebank POS tag to the tag WordNetLemmatizer expects."""
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


def lemmatize(tokens: List[str]) -> List[str]:
    """POS-aware lemmatization -- without the POS tag, WordNet assumes every
    word is a noun, so 'running'/'runs' never collapse to 'run'."""
    if not tokens:
        return []
    tagged = pos_tag(tokens)
    return [_LEMMATIZER.lemmatize(tok, pos=_wordnet_pos(tag)) for tok, tag in tagged]


def preprocess_text(text: str) -> ProcessedText:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    no_stop = remove_stopwords(tokens)
    lemmas = lemmatize(no_stop)
    processed = " ".join(lemmas)

    return ProcessedText(
        original_text=text if isinstance(text, str) else "",
        cleaned_text=cleaned,
        tokens=tokens,
        tokens_no_stopwords=no_stop,
        lemmatized_tokens=lemmas,
        processed_text=processed,
        is_empty_after_processing=(len(processed.strip()) == 0),
    )


if __name__ == "__main__":
    samples = [
        "I am SO happy today!!! Best day ever.",
        "   too      many     spaces   ",
        "!!!???...",
        "",
        "Running, ran, runs - the runner runs quickly.",
    ]
    for s in samples:
        r = preprocess_text(s)
        print(f"IN : {s!r}")
        print(f"OUT: {r.to_dict()}\n")
