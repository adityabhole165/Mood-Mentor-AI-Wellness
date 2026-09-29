"""Command-line controlled Task 9 evaluator.

Requires the trained Milestone 2 model and sentence-transformers assets used by
pipeline_v4. It never reads live feedback as ground truth.
"""
from __future__ import annotations
import argparse
import json
import os
from .evaluation_runner import load_evaluation_cases, run_controlled_evaluation
from .pipeline_v4 import run_milestone3_part2
from .recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/m3_evaluation_cases.json")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--model-type", choices=["BERT", "DistilBERT"], default="BERT")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    cases = load_evaluation_cases(args.dataset)
    by_id = {c.content_id: c for c in DEFAULT_WELLNESS_CONTENT}

    def advanced(case):
        result = run_milestone3_part2(
            case["text"], UserProfile(case["user_id"]),
            emotion_model_dir=args.model_dir, model_type=args.model_type,
            top_k=min(args.top_k, int(case["k"])), persist_history=False,
        )
        if not result.get("is_valid"):
            raise RuntimeError(result.get("error", "invalid evaluation case"))
        return [
            {"content_id": r["content_id"], "content_type": by_id.get(r["content_id"]).content_type if by_id.get(r["content_id"]) else None}
            for r in result["recommendations"]
        ]

    def baseline(case):
        result = run_milestone3_part2(
            case["text"], UserProfile(case["user_id"]),
            emotion_model_dir=args.model_dir, model_type=args.model_type,
            top_k=min(args.top_k, int(case["k"])), persist_history=False,
        )
        if not result.get("is_valid"):
            raise RuntimeError(result.get("error", "invalid evaluation case"))
        rows = sorted(result["candidates"], key=lambda x: (-x["rule_score"], x["content_id"]))[:min(args.top_k, int(case["k"]))]
        return [
            {"content_id": r["content_id"], "content_type": by_id.get(r["content_id"]).content_type if by_id.get(r["content_id"]) else None}
            for r in rows
        ]

    result = run_controlled_evaluation(cases, baseline, advanced)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
