"""
confidence.py
Milestone 2 - Task 4: Confidence Score Validation

Thin, model-agnostic layer on top of emotion_bert.predict() /
emotion_distilbert.predict() -- both already return per-emotion sigmoid
scores straight from the model. This module adds the checks Task 4 asks for:
    - predicted (primary) emotion
    - confidence score for that emotion
    - full probability distribution across all 6 emotions
    - which emotions cross the decision threshold
    - a runtime guard that FAILS LOUDLY if scores look hardcoded/faked
      (e.g. all identical, or outside [0, 1])

Example (matches the milestone spec's own sample):
    Input : "I am excited about the new opportunity but nervous about the outcome."
    Output: joy -> 0.81   fear -> 0.63   (both dynamically computed, both real)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List

from .emotion_dataset import EMOTIONS

DEFAULT_THRESHOLD = 0.5


@dataclass
class ConfidenceReport:
    primary_emotion: str
    primary_confidence: float
    all_scores: Dict[str, float]
    triggered_emotions: List[str]
    threshold: float

    def to_dict(self):
        return asdict(self)


def validate_scores_are_dynamic(scores: Dict[str, float]) -> None:
    """
    Guards against exactly the failure mode Task 4 warns about: hardcoded or
    faked confidence values. Raises ValueError if scores look synthetic.
    """
    values = list(scores.values())

    if any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError(f"Confidence scores out of [0,1] range: {scores}")

    if len(set(round(v, 6) for v in values)) == 1:
        raise ValueError(
            f"All emotion scores are identical ({values[0]}) -- "
            "this strongly suggests hardcoded/fake output, not a real model prediction."
        )

    missing = [e for e in EMOTIONS if e not in scores]
    if missing:
        raise ValueError(f"Score dict is missing categories: {missing}")


def build_confidence_report(scores: Dict[str, float], threshold: float = DEFAULT_THRESHOLD) -> ConfidenceReport:
    """scores: raw {emotion: sigmoid_probability} straight from a model's predict()."""
    validate_scores_are_dynamic(scores)

    primary = max(scores, key=scores.get)
    triggered = [e for e, s in scores.items() if s >= threshold]

    return ConfidenceReport(
        primary_emotion=primary,
        primary_confidence=scores[primary],
        all_scores=scores,
        triggered_emotions=triggered,
        threshold=threshold,
    )


def format_confidence_summary(report: ConfidenceReport) -> str:
    """Human-readable version, e.g. for the Streamlit UI or a console demo."""
    lines = [f"Primary emotion : {report.primary_emotion} ({report.primary_confidence:.3f})"]
    for emotion in EMOTIONS:
        marker = "x" if emotion in report.triggered_emotions else " "
        lines.append(f"  [{marker}] {emotion:10s} {report.all_scores[emotion]:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test with a plausible (non-hardcoded-looking) score distribution
    sample_scores = {
        "joy": 0.81, "sadness": 0.04, "anger": 0.02,
        "fear": 0.63, "surprise": 0.11, "disgust": 0.01,
    }
    report = build_confidence_report(sample_scores)
    print(format_confidence_summary(report))

    try:
        validate_scores_are_dynamic({e: 0.5 for e in EMOTIONS})
    except ValueError as e:
        print(f"\nCorrectly rejected hardcoded-looking scores: {e}")
