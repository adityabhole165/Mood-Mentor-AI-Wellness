
"""Milestone 4 Task 7 lightweight stress/performance runner.

This does not train models; it measures the deterministic recommendation
components so CI can detect regressions in throughput/latency.
"""
from __future__ import annotations
import time
from statistics import mean
from .recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile
from .emotional_state import analyze_emotional_state
from .hybrid_recommender import HybridRecommendationEngine
from .ranking import RecommendationRanker


def run_recommendation_stress(iterations: int = 25) -> dict:
    scores = {"joy": 0.8, "sadness": 0.05, "anger": 0.02, "fear": 0.08, "surprise": 0.03, "disgust": 0.02}
    state = analyze_emotional_state(scores, 0.6)
    profile = UserProfile("stress_user", [], [], set())
    engine = HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT, [], None, None)
    times = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        candidates = engine.generate_candidates(state, profile, {})
        RecommendationRanker().rank(candidates, state, top_k=5)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "iterations": len(times),
        "avg_ms": mean(times),
        "p95_ms": sorted(times)[max(0, int(len(times) * 0.95) - 1)],
        "max_ms": max(times),
    }


if __name__ == "__main__":
    print(run_recommendation_stress())
