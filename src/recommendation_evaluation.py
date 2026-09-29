"""Milestone 3 Task 9: reproducible recommendation evaluation."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Sequence


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

    def to_dict(self) -> dict:
        return asdict(self)


def _metrics(predicted: Sequence[str], relevant: set[str], k: int):
    top = list(predicted[:k])
    hits = sum(x in relevant for x in top)
    precision = hits / max(1, len(top))
    recall = hits / max(1, len(relevant))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(top) if item in relevant)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return precision, recall, f1, ndcg


def recommendation_diversity(recommendations: Sequence[dict]) -> float:
    if not recommendations:
        return 0.0
    types = [r.get("content_type") or r.get("components", {}).get("content_type") for r in recommendations]
    known = [x for x in types if x]
    if not known:
        ids = {r.get("content_id") for r in recommendations}
        return len(ids) / len(recommendations)
    return len(set(known)) / len(known)


def evaluate_rankings(
    system_name: str,
    predicted_lists: Sequence[Sequence[str]],
    relevant_sets: Sequence[set[str]],
    *,
    k: int = 5,
    accepted: Sequence[float | bool] | None = None,
    diversities: Sequence[float] | None = None,
    response_ms: Sequence[float] | None = None,
) -> RecommendationEvaluation:
    values = [_metrics(p, r, k) for p, r in zip(predicted_lists, relevant_sets)]
    p = [x[0] for x in values]
    r = [x[1] for x in values]
    f = [x[2] for x in values]
    n = [x[3] for x in values]
    return RecommendationEvaluation(
        system_name=system_name,
        precision_at_k=sum(p) / max(1, len(p)),
        recall_at_k=sum(r) / max(1, len(r)),
        f1_at_k=sum(f) / max(1, len(f)),
        ndcg_at_k=sum(n) / max(1, len(n)),
        acceptance_rate=sum(float(x) for x in accepted) / len(accepted) if accepted else 0.0,
        diversity=sum(diversities) / len(diversities) if diversities else 0.0,
        avg_response_ms=sum(response_ms) / len(response_ms) if response_ms else 0.0,
    )


def acceptance_for_predictions(predicted: Sequence[str], accepted_ids: set[str], k: int) -> float:
    """Controlled-dataset acceptance proxy: fraction of top-k items marked accepted."""
    top = list(predicted[:k])
    return sum(item in accepted_ids for item in top) / max(1, len(top))


def evaluate_recommender_pair(baseline_fn, advanced_fn, test_cases: Sequence[dict], *, k: int = 5) -> Dict[str, dict]:
    """Compare two recommenders on the same controlled cases.

    Each case must contain ``relevant_ids`` and may contain ``accepted_ids``.
    ``accepted_ids`` is the controlled ground truth used for the acceptance-rate
    metric; it is deliberately independent of live feedback CSV data.
    """
    outputs = {}
    for name, fn in (("baseline", baseline_fn), ("advanced_ml", advanced_fn)):
        predictions, relevant, accepted, times, diversities = [], [], [], [], []
        for case in test_cases:
            start = time.perf_counter()
            recommendations = fn(case)
            elapsed = (time.perf_counter() - start) * 1000.0
            predicted = [r["content_id"] for r in recommendations]
            predictions.append(predicted)
            relevant.append(set(case["relevant_ids"]))
            accepted.append(acceptance_for_predictions(predicted, set(case.get("accepted_ids", [])), k))
            times.append(elapsed)
            diversities.append(recommendation_diversity(recommendations))
        outputs[name] = evaluate_rankings(
            name, predictions, relevant, k=k, accepted=accepted,
            diversities=diversities, response_ms=times,
        ).to_dict()
    return outputs


def compare_baseline_and_advanced(results: Dict[str, dict]) -> dict:
    base, advanced = results["baseline"], results["advanced_ml"]
    metrics = ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k", "acceptance_rate", "diversity", "avg_response_ms"]
    delta = {metric: round(advanced[metric] - base[metric], 6) for metric in metrics}
    quality_metrics = ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k", "acceptance_rate", "diversity"]
    quality_improved = all(delta[m] >= 0 for m in quality_metrics) and any(delta[m] > 0 for m in quality_metrics)
    return {
        "baseline": base,
        "advanced_ml": advanced,
        "delta_advanced_minus_baseline": delta,
        "quality_improved": quality_improved,
        "performance_note": "Higher quality metrics are better. Lower avg_response_ms is faster.",
    }
