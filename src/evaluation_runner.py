"""Reusable controlled Task 9 evaluation runner."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable, Sequence
from .recommendation_evaluation import evaluate_recommender_pair, compare_baseline_and_advanced


def load_evaluation_cases(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a non-empty JSON list")
    required = {"case_id", "user_id", "text", "relevant_ids", "accepted_ids", "k"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"Evaluation case {case.get('case_id')} missing: {sorted(missing)}")
    return cases


def run_controlled_evaluation(
    cases: Sequence[dict],
    baseline_fn: Callable[[dict], list[dict]],
    advanced_fn: Callable[[dict], list[dict]],
) -> dict:
    if not cases:
        raise ValueError("No controlled evaluation cases supplied")
    k = max(int(case.get("k", 5)) for case in cases)
    results = evaluate_recommender_pair(baseline_fn, advanced_fn, cases, k=k)
    return compare_baseline_and_advanced(results)
