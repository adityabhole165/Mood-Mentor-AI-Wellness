"""Milestone 3 Part 2 - Task 9: recommendation evaluation and performance tests."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence


@dataclass
class RecommendationEvaluation:
    system_name: str
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    ndcg_at_k: float
    acceptance_rate: float
    diversity: float
    avg_response_ms: float

    def to_dict(self):
        return asdict(self)


def _metrics(predicted: Sequence[str], relevant: set[str], k: int) -> tuple[float, float, float, float]:
    top = list(predicted[:k])
    hits = sum(x in relevant for x in top)
    precision = hits / max(1, len(top))
    recall = hits / max(1, len(relevant))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(top)
        if item in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, f1, ndcg


def recommendation_diversity(recommendations: Sequence[dict]) -> float:
    """1 - duplicate ratio, using content type as a simple diversity proxy."""
    if not recommendations:
        return 0.0
    types = [
        r.get("content_type") or r.get("components", {}).get("content_type")
        for r in recommendations
    ]
    known = [x for x in types if x]
    if not known:
        return len({r.get("content_id") for r in recommendations}) / len(recommendations)
    return len(set(known)) / len(known)


def evaluate_rankings(
    system_name: str,
    predicted_lists: Sequence[Sequence[str]],
    relevant_sets: Sequence[set[str]],
    *,
    k: int = 5,
    accepted: Sequence[bool] | None = None,
    diversities: Sequence[float] | None = None,
    response_ms: Sequence[float] | None = None,
) -> RecommendationEvaluation:
    p, r, f, n = [], [], [], []
    for pred, rel in zip(predicted_lists, relevant_sets):
        a, b, c, d = _metrics(pred, rel, k)
        p.append(a); r.append(b); f.append(c); n.append(d)

    return RecommendationEvaluation(
        system_name=system_name,
        precision_at_k=sum(p) / max(1, len(p)),
        recall_at_k=sum(r) / max(1, len(r)),
        f1_at_k=sum(f) / max(1, len(f)),
        ndcg_at_k=sum(n) / max(1, len(n)),
        acceptance_rate=(sum(accepted) / len(accepted)) if accepted else 0.0,
        diversity=(sum(diversities) / len(diversities)) if diversities else 0.0,
        avg_response_ms=(sum(response_ms) / len(response_ms)) if response_ms else 0.0,
    )


def evaluate_recommender_pair(
    baseline_fn,
    advanced_fn,
    test_cases: Sequence[dict],
    *,
    k: int = 5,
) -> Dict[str, dict]:
    """
    test_cases format:
      {"input": ..., "relevant_ids": set/list, "user_id": ...}

    baseline_fn(case) and advanced_fn(case) must return recommendation dicts
    containing content_id. The functions are timed independently.
    """
    outputs = {}
    for name, fn in (("baseline", baseline_fn), ("advanced_ml", advanced_fn)):
        predictions, relevant, times, diversities = [], [], [], []
        for case in test_cases:
            start = time.perf_counter()
            recs = fn(case)
            elapsed = (time.perf_counter() - start) * 1000
            predictions.append([r["content_id"] for r in recs])
            relevant.append(set(case["relevant_ids"]))
            times.append(elapsed)
            diversities.append(recommendation_diversity(recs))
        outputs[name] = evaluate_rankings(
            name, predictions, relevant, k=k,
            diversities=diversities, response_ms=times
        ).to_dict()
    return outputs


def compare_baseline_and_advanced(results: Dict[str, dict]) -> dict:
    """Report metric deltas; does not hide cases where advanced ML regresses."""
    base = results["baseline"]
    adv = results["advanced_ml"]
    metrics = [
        "precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k",
        "acceptance_rate", "diversity", "avg_response_ms"
    ]
    delta = {m: round(adv[m] - base[m], 6) for m in metrics}
    return {
        "baseline": base,
        "advanced_ml": adv,
        "delta_advanced_minus_baseline": delta,
        "quality_improved": all(
            delta[m] >= 0 for m in
            ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"]
        ),
        "performance_note": (
            "Lower avg_response_ms is faster."
        ),
    }
