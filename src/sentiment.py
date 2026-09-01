"""
sentiment.py
Milestone 1 - Task 3: VADER Sentiment Validation

Baseline sentiment scoring using VADER (Valence Aware Dictionary and sEntiment
Reasoner) -- a lexicon + rule-based model tuned for short, informal, emoji- and
punctuation-heavy text (chat messages, social posts). That's exactly why it's
the right baseline before Milestone 2's transformer-based emotion classifier:
it needs no training data and gives you a working end-to-end pipeline on day 1.

Every score below comes directly from analyzer.polarity_scores() -- nothing
here is hardcoded or faked.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYZER = SentimentIntensityAnalyzer()

# Standard VADER thresholds (from the VADER paper / reference implementation)
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


@dataclass
class SentimentResult:
    text: str
    compound: float
    pos: float
    neg: float
    neu: float
    label: str  # "positive" | "negative" | "neutral"

    def to_dict(self):
        return asdict(self)


def classify_compound(compound: float) -> str:
    """Pure function -- easy to unit test in isolation."""
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Feed the CLEANED text here (preprocessing.clean_text), NOT the fully
    tokenized/lemmatized/stopword-stripped version -- VADER needs case,
    punctuation, and emojis intact to score correctly.
    """
    if not text or not text.strip():
        text = ""

    scores = _ANALYZER.polarity_scores(text)
    return SentimentResult(
        text=text,
        compound=scores["compound"],
        pos=scores["pos"],
        neg=scores["neg"],
        neu=scores["neu"],
        label=classify_compound(scores["compound"]),
    )


if __name__ == "__main__":
    samples = [
        "I absolutely love this! Best day ever!!",
        "This is the worst experience I've ever had. Terrible.",
        "The meeting is scheduled for 3pm tomorrow.",
        "I'm not sure how I feel about this, kind of mixed.",
        "AMAZING!!! :) :) :)",
        "I hate waiting in line so much",
    ]
    for s in samples:
        r = analyze_sentiment(s)
        print(f"{r.label.upper():9s} compound={r.compound:+.3f}  pos={r.pos:.3f} neg={r.neg:.3f} neu={r.neu:.3f}  | {s}")
