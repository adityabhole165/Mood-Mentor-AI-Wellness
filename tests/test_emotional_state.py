import pytest

from src.emotional_state import (
    analyze_emotional_state,
    clamp,
    normalized_entropy,
)


# ============================================================
# TEST 1: Basic positive emotion
# ============================================================

def test_positive_emotional_state():

    emotion_scores = {
        "joy": 0.90,
        "sadness": 0.05,
        "anger": 0.02,
        "fear": 0.01,
        "surprise": 0.01,
        "disgust": 0.01,
    }

    result = analyze_emotional_state(
        emotion_scores=emotion_scores,
        polarity_score=0.80,
    )

    assert result.dominant_emotion == "joy"

    assert result.emotion_confidence == 0.90

    assert result.polarity == "positive"

    assert result.polarity_score == 0.80

    assert result.intensity > 0

    assert result.positive_emotion_load > 0

    assert result.negative_emotion_load >= 0


# ============================================================
# TEST 2: Strong negative emotional state
# ============================================================

def test_negative_emotional_state():

    emotion_scores = {
        "joy": 0.01,
        "sadness": 0.85,
        "anger": 0.20,
        "fear": 0.75,
        "surprise": 0.02,
        "disgust": 0.05,
    }

    result = analyze_emotional_state(
        emotion_scores=emotion_scores,
        polarity_score=-0.90,
    )

    assert result.dominant_emotion == "sadness"

    assert result.polarity == "negative"

    assert result.polarity_score == -0.90

    assert result.intensity > 0.50

    assert result.negative_emotion_load > 0

    assert result.severity in {
        "moderate",
        "high",
        "critical",
    }


# ============================================================
# TEST 3: Neutral polarity
# ============================================================

def test_neutral_polarity():

    emotion_scores = {
        "joy": 0.10,
        "sadness": 0.10,
        "anger": 0.05,
        "fear": 0.05,
        "surprise": 0.20,
        "disgust": 0.05,
    }

    result = analyze_emotional_state(
        emotion_scores=emotion_scores,
        polarity_score=0.0,
    )

    assert result.polarity == "neutral"

    assert result.polarity_score == 0.0

    assert 0.0 <= result.intensity <= 1.0


# ============================================================
# TEST 4: Multiple emotions
# ============================================================

def test_multiple_triggered_emotions():

    emotion_scores = {
        "joy": 0.70,
        "sadness": 0.60,
        "anger": 0.10,
        "fear": 0.05,
        "surprise": 0.10,
        "disgust": 0.02,
    }

    result = analyze_emotional_state(
        emotion_scores=emotion_scores,
        polarity_score=0.10,
        trigger_threshold=0.35,
    )

    assert "joy" in result.triggered_emotions

    assert "sadness" in result.triggered_emotions

    assert len(result.triggered_emotions) >= 2


# ============================================================
# TEST 5: Mixed emotional state
# ============================================================

def test_mixed_emotional_state():

    emotion_scores = {
        "joy": 0.80,
        "sadness": 0.75,
        "anger": 0.05,
        "fear": 0.05,
        "surprise": 0.05,
        "disgust": 0.02,
    }

    result = analyze_emotional_state(
        emotion_scores=emotion_scores,
        polarity_score=0.0,
        trigger_threshold=0.35,
    )

    assert result.mixed_state is True

    assert "joy" in result.triggered_emotions

    assert "sadness" in result.triggered_emotions


# ============================================================
# TEST 6: Low emotional intensity
# ============================================================

def test_low_intensity():

    emotion_scores = {
        "joy": 0.10,
        "sadness": 0.05,
        "anger": 0.02,
        "fear": 0.03,
        "surprise": 0.02,
        "disgust": 0.01,
    }

    result = analyze_emotional_state(
        emotion_scores=emotion_scores,
        polarity_score=0.02,
    )

    assert 0.0 <= result.intensity <= 1.0

    assert result.intensity < 0.50

    assert result.severity == "low"


# ============================================================
# TEST 7: High emotional intensity
# ============================================================

def test_high_intensity():

    emotion_scores = {
        "joy": 0.01,
        "sadness": 0.92,
        "anger": 0.10,
        "fear": 0.85,
        "surprise": 0.02,
        "disgust": 0.05,
    }

    result = analyze_emotional_state(
        emotion_scores=emotion_scores,
        polarity_score=-0.90,
    )

    assert 0.0 <= result.intensity <= 1.0

    assert result.intensity > 0.50

    assert result.severity in {
        "moderate",
        "high",
        "critical",
    }


# ============================================================
# TEST 8: Intensity is dynamic
# ============================================================

def test_intensity_is_dynamic():

    low_emotion = {
        "joy": 0.10,
        "sadness": 0.05,
        "anger": 0.02,
        "fear": 0.03,
        "surprise": 0.02,
        "disgust": 0.01,
    }

    high_emotion = {
        "joy": 0.01,
        "sadness": 0.90,
        "anger": 0.20,
        "fear": 0.85,
        "surprise": 0.02,
        "disgust": 0.05,
    }

    low_result = analyze_emotional_state(
        low_emotion,
        polarity_score=0.02,
    )

    high_result = analyze_emotional_state(
        high_emotion,
        polarity_score=-0.90,
    )

    assert high_result.intensity > low_result.intensity


# ============================================================
# TEST 9: Emotion probabilities are clamped
# ============================================================

def test_probability_clamping():

    emotion_scores = {
        "joy": 1.50,
        "sadness": -0.50,
        "anger": 0.20,
        "fear": 0.10,
        "surprise": 0.05,
        "disgust": 0.02,
    }

    result = analyze_emotional_state(
        emotion_scores,
        polarity_score=0.50,
    )

    assert result.emotion_probabilities["joy"] == 1.0

    assert result.emotion_probabilities["sadness"] == 0.0


# ============================================================
# TEST 10: Polarity score is clamped
# ============================================================

def test_polarity_clamping():

    emotion_scores = {
        "joy": 0.80,
        "sadness": 0.05,
        "anger": 0.02,
        "fear": 0.01,
        "surprise": 0.01,
        "disgust": 0.01,
    }

    positive_result = analyze_emotional_state(
        emotion_scores,
        polarity_score=2.0,
    )

    negative_result = analyze_emotional_state(
        emotion_scores,
        polarity_score=-2.0,
    )

    assert positive_result.polarity_score == 1.0

    assert negative_result.polarity_score == -1.0


# ============================================================
# TEST 11: Uncertainty
# ============================================================

def test_uncertainty():

    emotion_scores = {
        "joy": 0.90,
        "sadness": 0.05,
        "anger": 0.02,
        "fear": 0.01,
        "surprise": 0.01,
        "disgust": 0.01,
    }

    result = analyze_emotional_state(
        emotion_scores,
        polarity_score=0.80,
    )

    expected_uncertainty = 1.0 - 0.90

    assert result.uncertainty == expected_uncertainty


# ============================================================
# TEST 12: Empty emotion scores should fail
# ============================================================

def test_empty_emotion_scores():

    with pytest.raises(ValueError):

        analyze_emotional_state(
            emotion_scores={},
            polarity_score=0.0,
        )


# ============================================================
# TEST 13: Trigger threshold works
# ============================================================

def test_trigger_threshold():

    emotion_scores = {
        "joy": 0.60,
        "sadness": 0.40,
        "anger": 0.20,
        "fear": 0.10,
        "surprise": 0.05,
        "disgust": 0.02,
    }

    result = analyze_emotional_state(
        emotion_scores,
        polarity_score=0.10,
        trigger_threshold=0.50,
    )

    assert "joy" in result.triggered_emotions

    assert "sadness" not in result.triggered_emotions


# ============================================================
# TEST 14: to_dict() works
# ============================================================

def test_to_dict():

    emotion_scores = {
        "joy": 0.80,
        "sadness": 0.10,
        "anger": 0.05,
        "fear": 0.02,
        "surprise": 0.02,
        "disgust": 0.01,
    }

    result = analyze_emotional_state(
        emotion_scores,
        polarity_score=0.70,
    )

    data = result.to_dict()

    assert isinstance(data, dict)

    assert data["dominant_emotion"] == "joy"

    assert "intensity" in data

    assert "polarity" in data

    assert "severity" in data


# ============================================================
# TEST 15: clamp() utility
# ============================================================

def test_clamp():

    assert clamp(-10) == 0.0

    assert clamp(0.5) == 0.5

    assert clamp(10) == 1.0


# ============================================================
# TEST 16: normalized entropy
# ============================================================

def test_normalized_entropy():

    # One emotion = very concentrated
    low_entropy = normalized_entropy(
        [1.0, 0.0, 0.0]
    )

    # Equal emotions = high entropy
    high_entropy = normalized_entropy(
        [1.0, 1.0, 1.0]
    )

    assert low_entropy == 0.0

    assert high_entropy == pytest.approx(1.0)