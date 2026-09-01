"""Task 3 - VADER Sentiment Validation"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentiment import analyze_sentiment, classify_compound, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD


def test_positive_sentiment_detected():
    r = analyze_sentiment("I absolutely love this, best day ever!")
    assert r.label == "positive"
    assert r.compound >= POSITIVE_THRESHOLD

def test_negative_sentiment_detected():
    r = analyze_sentiment("This is terrible, I hate everything about it.")
    assert r.label == "negative"
    assert r.compound <= NEGATIVE_THRESHOLD

def test_neutral_sentiment_detected():
    r = analyze_sentiment("The report is due on Friday.")
    assert r.label == "neutral"

def test_compound_score_within_valid_range():
    r = analyze_sentiment("I'm feeling okay about things today.")
    assert -1.0 <= r.compound <= 1.0

def test_pos_neg_neu_scores_sum_to_approximately_one():
    r = analyze_sentiment("I am so happy and grateful for this wonderful day!")
    assert abs((r.pos + r.neg + r.neu) - 1.0) < 0.01

def test_different_inputs_give_different_scores_not_hardcoded():
    r1 = analyze_sentiment("I love this so much!")
    r2 = analyze_sentiment("I hate this so much!")
    r3 = analyze_sentiment("The train arrives at 9am.")
    scores = {r1.compound, r2.compound, r3.compound}
    assert len(scores) == 3  # all three are distinct -> nothing is hardcoded
    assert r1.compound > 0
    assert r2.compound < 0

def test_classify_compound_boundaries():
    assert classify_compound(0.05) == "positive"
    assert classify_compound(0.049) == "neutral"
    assert classify_compound(-0.05) == "negative"
    assert classify_compound(-0.049) == "neutral"
    assert classify_compound(0.0) == "neutral"

def test_empty_text_returns_neutral_without_crash():
    r = analyze_sentiment("")
    assert r.label == "neutral"
    assert r.compound == 0.0

def test_emoji_and_punctuation_affect_score():
    plain = analyze_sentiment("I am happy")
    emphasized = analyze_sentiment("I am SO happy!!!")
    # capitalization + punctuation should push the compound score higher
    assert emphasized.compound >= plain.compound

def test_sentiment_result_has_all_required_fields():
    r = analyze_sentiment("Testing all fields present")
    d = r.to_dict()
    for field in ["text", "compound", "pos", "neg", "neu", "label"]:
        assert field in d
