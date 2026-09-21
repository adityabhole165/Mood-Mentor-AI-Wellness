"""Data contracts used by the Milestone 3 recommendation system."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set

@dataclass
class WellnessContent:
    content_id: str
    title: str
    description: str
    content_type: str
    tags: List[str]
    emotions: List[str]
    url: str = ""
    duration_minutes: int = 5
    intensity_level: int = 1

    def to_dict(self):
        return asdict(self)

@dataclass
class UserProfile:
    user_id: str
    preferred_types: List[str] = field(default_factory=list)
    preferred_tags: List[str] = field(default_factory=list)
    blocked_content_ids: Set[str] = field(default_factory=set)

@dataclass
class Interaction:
    user_id: str
    content_id: str
    reward: float
    interaction_type: str = ""
    timestamp: str = ""
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    emotion_intensity: float = 0.0
    feature_vector: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

DEFAULT_WELLNESS_CONTENT = [
    WellnessContent("W001", "2-Minute Box Breathing", "A short guided breathing exercise for calming an activated state.", "breathing", ["breathing", "calm", "stress"], ["fear", "anger", "sadness"], duration_minutes=2, intensity_level=1),
    WellnessContent("W002", "5-4-3-2-1 Grounding", "A sensory grounding exercise to bring attention back to the present.", "grounding", ["grounding", "anxiety", "present"], ["fear", "sadness"], duration_minutes=5, intensity_level=2),
    WellnessContent("W003", "Guided Mood Journal", "A short reflection prompt for identifying feelings and next steps.", "journal", ["reflection", "journaling", "self-awareness"], ["sadness", "anger", "surprise"], duration_minutes=10, intensity_level=2),
    WellnessContent("W004", "Three Good Things", "A brief gratitude activity focused on noticing positive experiences.", "reflection", ["gratitude", "positivity", "reflection"], ["joy", "sadness"], duration_minutes=5, intensity_level=1),
    WellnessContent("W005", "Pause Before Reacting", "A structured pause-and-reflect exercise for strong anger.", "exercise", ["anger-management", "pause", "reflection"], ["anger"], duration_minutes=7, intensity_level=3),
    WellnessContent("W006", "Wind-Down Routine", "A gentle evening routine designed to reduce mental activation.", "routine", ["sleep", "relaxation", "routine"], ["fear", "sadness"], duration_minutes=10, intensity_level=2),
    WellnessContent("W007", "Positive Momentum Activity", "A small action that helps maintain a positive emotional state.", "activity", ["positivity", "habits", "motivation"], ["joy", "surprise"], duration_minutes=8, intensity_level=1),
]
