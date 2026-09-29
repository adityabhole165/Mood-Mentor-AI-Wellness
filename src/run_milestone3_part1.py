"""CLI runner for Milestone 3 Part 1 (Tasks 1-5)."""
from __future__ import annotations
import argparse
import json
from .pipeline_v3 import run_milestone3
from .recommendation_data import UserProfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MoodMentor Milestone 3 Part 1.")
    parser.add_argument("--text", required=True, help="User text to analyze.")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-dir", default=None, help="Milestone 2 trained BERT model directory.")
    parser.add_argument("--semantic-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    profile = UserProfile(user_id=args.user_id)
    result = run_milestone3(
        args.text,
        profile,
        emotion_model_dir=args.model_dir,
        top_k=args.top_k,
        semantic_model_name=args.semantic_model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("is_valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
