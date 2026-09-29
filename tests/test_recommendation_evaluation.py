import math

import pytest

from src.recommendation_evaluation import (
    RecommendationEvaluation,
    _metrics,
    recommendation_diversity,
    evaluate_rankings,
    evaluate_recommender_pair,
    compare_baseline_and_advanced,
)


def test_metrics_perfect_prediction():
    predicted = ["A", "B", "C"]
    relevant = {"A", "B", "C"}

    precision, recall, f1, ndcg = _metrics(
        predicted,
        relevant,
        k=3,
    )

    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    assert ndcg == 1.0


def test_metrics_no_hits():
    predicted = ["A", "B", "C"]
    relevant = {"X", "Y"}

    precision, recall, f1, ndcg = _metrics(
        predicted,
        relevant,
        k=3,
    )

    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0
    assert ndcg == 0.0


def test_metrics_partial_prediction():
    predicted = ["A", "B", "C"]
    relevant = {"A", "X"}

    precision, recall, f1, ndcg = _metrics(
        predicted,
        relevant,
        k=3,
    )

    assert precision == pytest.approx(1 / 3)
    assert recall == pytest.approx(1 / 2)
    assert f1 == pytest.approx(2 * (1 / 3) * (1 / 2) / ((1 / 3) + (1 / 2)))
    assert 0 < ndcg < 1


def test_metrics_respects_k():
    predicted = ["A", "B", "C"]
    relevant = {"C"}

    precision, recall, f1, ndcg = _metrics(
        predicted,
        relevant,
        k=1,
    )

    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0
    assert ndcg == 0.0


def test_recommendation_diversity_with_unique_types():
    recommendations = [
        {"content_id": "A", "content_type": "breathing"},
        {"content_id": "B", "content_type": "meditation"},
        {"content_id": "C", "content_type": "journal"},
    ]

    assert recommendation_diversity(recommendations) == 1.0


def test_recommendation_diversity_with_duplicates():
    recommendations = [
        {"content_id": "A", "content_type": "breathing"},
        {"content_id": "B", "content_type": "breathing"},
        {"content_id": "C", "content_type": "meditation"},
    ]

    assert recommendation_diversity(recommendations) == pytest.approx(2 / 3)


def test_recommendation_diversity_empty():
    assert recommendation_diversity([]) == 0.0


def test_recommendation_diversity_uses_content_id_when_type_missing():
    recommendations = [
        {"content_id": "A"},
        {"content_id": "B"},
        {"content_id": "A"},
    ]

    assert recommendation_diversity(recommendations) == pytest.approx(2 / 3)


def test_evaluate_rankings():
    predicted = [
        ["A", "B", "C"],
        ["X", "Y", "Z"],
    ]

    relevant = [
        {"A", "C"},
        {"X"},
    ]

    result = evaluate_rankings(
        "test-system",
        predicted,
        relevant,
        k=3,
        accepted=[True, False],
        diversities=[1.0, 0.5],
        response_ms=[10.0, 20.0],
    )

    assert isinstance(result, RecommendationEvaluation)
    assert result.system_name == "test-system"
    assert result.precision_at_k > 0
    assert result.recall_at_k > 0
    assert result.f1_at_k > 0
    assert result.ndcg_at_k > 0
    assert result.acceptance_rate == 0.5
    assert result.diversity == 0.75
    assert result.avg_response_ms == 15.0


def test_evaluate_rankings_empty_optional_metrics():
    result = evaluate_rankings(
        "test",
        [["A"]],
        [{"A"}],
    )

    assert result.acceptance_rate == 0.0
    assert result.diversity == 0.0
    assert result.avg_response_ms == 0.0


def test_evaluate_recommender_pair():
    cases = [
        {
            "input": "case 1",
            "relevant_ids": {"A"},
            "user_id": "u1",
        },
        {
            "input": "case 2",
            "relevant_ids": {"B"},
            "user_id": "u2",
        },
    ]

    def baseline_fn(case):
        if case["user_id"] == "u1":
            return [
                {"content_id": "A", "content_type": "breathing"},
            ]

        return [
            {"content_id": "X", "content_type": "meditation"},
        ]

    def advanced_fn(case):
        if case["user_id"] == "u1":
            return [
                {"content_id": "A", "content_type": "breathing"},
            ]

        return [
            {"content_id": "B", "content_type": "meditation"},
        ]

    results = evaluate_recommender_pair(
        baseline_fn,
        advanced_fn,
        cases,
        k=1,
    )

    assert "baseline" in results
    assert "advanced_ml" in results

    assert results["baseline"]["precision_at_k"] == 0.5
    assert results["advanced_ml"]["precision_at_k"] == 1.0


def test_compare_baseline_and_advanced():
    results = {
        "baseline": {
            "precision_at_k": 0.50,
            "recall_at_k": 0.40,
            "f1_at_k": 0.44,
            "ndcg_at_k": 0.60,
            "acceptance_rate": 0.50,
            "diversity": 0.50,
            "avg_response_ms": 20.0,
        },
        "advanced_ml": {
            "precision_at_k": 0.70,
            "recall_at_k": 0.60,
            "f1_at_k": 0.64,
            "ndcg_at_k": 0.75,
            "acceptance_rate": 0.70,
            "diversity": 0.80,
            "avg_response_ms": 15.0,
        },
    }

    comparison = compare_baseline_and_advanced(results)

    assert comparison["delta_advanced_minus_baseline"]["precision_at_k"] == 0.2
    assert comparison["delta_advanced_minus_baseline"]["recall_at_k"] == 0.2
    assert comparison["delta_advanced_minus_baseline"]["ndcg_at_k"] == 0.15

    assert comparison["quality_improved"] is True
    assert "Lower avg_response_ms is faster." in comparison["performance_note"]