from src.evaluation_runner import load_evaluation_cases, run_controlled_evaluation


def _baseline(case):
    return [{"content_id": x} for x in case["relevant_ids"][:case["k"]]]


def _advanced(case):
    return [{"content_id": x} for x in case["accepted_ids"] + [x for x in case["relevant_ids"] if x not in case["accepted_ids"]]]


def test_controlled_dataset_and_acceptance_metric():
    cases = load_evaluation_cases("data/m3_evaluation_cases.json")
    assert len(cases) == 6
    result = run_controlled_evaluation(cases, _baseline, _advanced)
    assert 0.0 <= result["baseline"]["acceptance_rate"] <= 1.0
    assert 0.0 <= result["advanced_ml"]["acceptance_rate"] <= 1.0
    assert result["advanced_ml"]["acceptance_rate"] >= result["baseline"]["acceptance_rate"]
