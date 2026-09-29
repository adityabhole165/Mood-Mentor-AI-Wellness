
from src.regression_tests import run_regression_tests
from src.recommendation_evaluation import (
    _metrics,
    recommendation_diversity,
    evaluate_rankings,
    compare_baseline_and_advanced,
)


def test_milestone3_part2_regression_suite():
    result = run_regression_tests()

    assert result["passed"] is True
    assert result["checks"] == 8


def test_task9_metrics_perfect_ranking():
    precision, recall, f1, ndcg = _metrics(
        ["W001", "W002", "W003"],
        {"W001", "W002", "W003"},
        3,
    )

    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    assert ndcg == 1.0


def test_task9_diversity():
    recommendations = [
        {"content_id": "W001", "content_type": "breathing"},
        {"content_id": "W002", "content_type": "grounding"},
        {"content_id": "W003", "content_type": "journal"},
    ]

    assert recommendation_diversity(recommendations) == 1.0


def test_task9_evaluation():
    result = evaluate_rankings(
        "milestone3_part2",
        [["W001", "W002"], ["W003", "W004"]],
        [{"W001"}, {"W003"}],
        k=2,
        accepted=[True, False],
        diversities=[1.0, 0.5],
        response_ms=[10.0, 20.0],
    )

    assert result.system_name == "milestone3_part2"
    assert result.precision_at_k == 0.5
    assert result.recall_at_k == 1.0
    assert result.acceptance_rate == 0.5
    assert result.diversity == 0.75
    assert result.avg_response_ms == 15.0


def test_task9_comparison_reports_deltas():
    results = {
        "baseline": {
            "precision_at_k": 0.4,
            "recall_at_k": 0.4,
            "f1_at_k": 0.4,
            "ndcg_at_k": 0.4,
            "acceptance_rate": 0.4,
            "diversity": 0.5,
            "avg_response_ms": 20.0,
        },
        "advanced_ml": {
            "precision_at_k": 0.6,
            "recall_at_k": 0.6,
            "f1_at_k": 0.6,
            "ndcg_at_k": 0.6,
            "acceptance_rate": 0.5,
            "diversity": 0.7,
            "avg_response_ms": 15.0,
        },
    }

    comparison = compare_baseline_and_advanced(results)

    assert (
        comparison["delta_advanced_minus_baseline"]["precision_at_k"]
        == 0.2
    )

    assert (
        comparison["delta_advanced_minus_baseline"]["ndcg_at_k"]
        == 0.2
    )

    assert comparison["quality_improved"] is True

