"""
emotional_state.py

Milestone 3 - Task 1
Deep Emotional State Analysis

Input:
    - emotion probabilities from BERT/DistilBERT
    - VADER polarity score

Output:
    - dominant emotion
    - multiple emotions
    - confidence
    - intensity
    - positive/negative polarity
    - mixed emotional state
    - emotional severity
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List
import math


# ---------------------------------------------------------------------------
# Emotion groups
# ---------------------------------------------------------------------------

NEGATIVE_EMOTIONS = {
    "sadness",
    "anger",
    "fear",
    "disgust",
}

POSITIVE_EMOTIONS = {
    "joy",
}

NEUTRAL_EMOTIONS = {
    "surprise",
}


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class EmotionalState:

    # Dominant emotion
    dominant_emotion: str
    emotion_confidence: float

    # All model probabilities
    emotion_probabilities: Dict[str, float]

    # Multiple detected emotions
    triggered_emotions: List[str]

    # Dynamic intensity
    intensity: float

    # Sentiment polarity
    polarity: str
    polarity_score: float

    # Mixed state
    mixed_state: bool

    # Severity
    severity: str

    # Supporting metrics
    negative_emotion_load: float
    positive_emotion_load: float
    uncertainty: float

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clamp(value: float) -> float:
    """
    Keep a numeric value between 0 and 1.
    """
    return max(0.0, min(1.0, float(value)))


def normalized_entropy(values: List[float]) -> float:
    """
    Calculate normalized entropy.

    Higher entropy:
        emotions are distributed across multiple classes.

    Lower entropy:
        one emotion dominates.
    """

    if not values:
        return 0.0

    total = sum(values)

    if total <= 0:
        return 0.0

    probabilities = [
        value / total
        for value in values
        if value > 0
    ]

    if len(probabilities) <= 1:
        return 0.0

    entropy = -sum(
        p * math.log(p)
        for p in probabilities
    )

    maximum_entropy = math.log(len(probabilities))

    if maximum_entropy == 0:
        return 0.0

    return entropy / maximum_entropy


# ---------------------------------------------------------------------------
# Main emotional state function
# ---------------------------------------------------------------------------

def analyze_emotional_state(
    emotion_scores: Dict[str, float],
    polarity_score: float,
    trigger_threshold: float = 0.35,
) -> EmotionalState:
    """
    Convert model probabilities + VADER polarity into a richer emotional state.

    Intensity is dynamically calculated from:

        dominant emotion confidence
        +
        emotional load
        +
        polarity magnitude
        +
        emotional concentration
    """

    if not emotion_scores:
        raise ValueError(
            "emotion_scores cannot be empty"
        )

    # Clean probabilities
    scores = {
        emotion: clamp(score)
        for emotion, score in emotion_scores.items()
    }

    # -----------------------------------------------------------------------
    # Dominant emotion
    # -----------------------------------------------------------------------

    dominant_emotion = max(
        scores,
        key=scores.get
    )

    dominant_confidence = scores[
        dominant_emotion
    ]

    # -----------------------------------------------------------------------
    # Multiple emotions
    # -----------------------------------------------------------------------

    triggered_emotions = [
        emotion
        for emotion, score in scores.items()
        if score >= trigger_threshold
    ]

    # -----------------------------------------------------------------------
    # Positive / negative emotional load
    # -----------------------------------------------------------------------

    positive_load = sum(
        scores.get(emotion, 0.0)
        for emotion in POSITIVE_EMOTIONS
    )

    negative_load = sum(
        scores.get(emotion, 0.0)
        for emotion in NEGATIVE_EMOTIONS
    )

    positive_load = clamp(
        positive_load
    )

    # Normalize negative load because there are several
    # negative emotion categories.
    negative_load_normalized = clamp(
        negative_load / max(
            1,
            len(NEGATIVE_EMOTIONS)
        )
    )

    # -----------------------------------------------------------------------
    # Polarity
    # -----------------------------------------------------------------------

    polarity_score = float(
        max(-1.0, min(1.0, polarity_score))
    )

    if polarity_score > 0.05:
        polarity = "positive"

    elif polarity_score < -0.05:
        polarity = "negative"

    else:
        polarity = "neutral"

    polarity_magnitude = clamp(
        abs(polarity_score)
    )

    # -----------------------------------------------------------------------
    # Emotion concentration
    # -----------------------------------------------------------------------

    entropy = normalized_entropy(
        list(scores.values())
    )

    concentration = clamp(
        1.0 - entropy
    )

    # -----------------------------------------------------------------------
    # DYNAMIC INTENSITY
    # -----------------------------------------------------------------------
    #
    # This is intentionally calculated at runtime.
    #
    # The intensity depends on:
    #
    #   40% dominant confidence
    #   25% emotional load
    #   20% sentiment polarity magnitude
    #   15% emotion concentration
    #
    # Therefore different model outputs produce different intensities.
    # -----------------------------------------------------------------------

    intensity = clamp(
        0.40 * dominant_confidence
        +
        0.25 * max(
            positive_load,
            negative_load
        )
        +
        0.20 * polarity_magnitude
        +
        0.15 * concentration
    )

    # -----------------------------------------------------------------------
    # Mixed emotional state
    # -----------------------------------------------------------------------

    mixed_state = (
        len(triggered_emotions) >= 2
        or (
            positive_load >= trigger_threshold
            and
            negative_load_normalized >= trigger_threshold
        )
    )

    # -----------------------------------------------------------------------
    # Emotional severity
    # -----------------------------------------------------------------------

    # Negative states receive stronger severity weighting.
    negative_risk = clamp(
        0.65 * intensity
        +
        0.35 * negative_load_normalized
    )

    if negative_risk >= 0.80:
        severity = "critical"

    elif negative_risk >= 0.60:
        severity = "high"

    elif negative_risk >= 0.35:
        severity = "moderate"

    else:
        severity = "low"

    # -----------------------------------------------------------------------
    # Uncertainty
    # -----------------------------------------------------------------------

    uncertainty = clamp(
        1.0 - dominant_confidence
    )

    return EmotionalState(

        dominant_emotion=dominant_emotion,

        emotion_confidence=dominant_confidence,

        emotion_probabilities=scores,

        triggered_emotions=triggered_emotions,

        intensity=intensity,

        polarity=polarity,

        polarity_score=polarity_score,

        mixed_state=mixed_state,

        severity=severity,

        negative_emotion_load=negative_load_normalized,

        positive_emotion_load=positive_load,

        uncertainty=uncertainty,
    )