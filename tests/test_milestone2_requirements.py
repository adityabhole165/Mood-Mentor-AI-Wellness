
import pytest


def test_emotion_categories_defined():
    from src.emotion_dataset import EMOTIONS
    assert set(EMOTIONS) == {"joy","sadness","anger","fear","surprise","disgust"}


def test_confidence_is_dynamic():
    from src.confidence import build_confidence_report
    report = build_confidence_report({"joy":0.9,"sadness":0.1,"anger":0.0,"fear":0.0,"surprise":0.0,"disgust":0.0})
    assert report.primary_emotion == "joy"
    assert report.primary_confidence == 0.9
