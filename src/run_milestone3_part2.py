"""CLI runner for Milestone 3 Part 2 (Tasks 6-10)."""
from __future__ import annotations
import argparse
import json
from .pipeline_v4 import run_milestone3_part2
from .recommendation_data import UserProfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MoodMentor Milestone 3 Part 2.")
    parser.add_argument("--text", required=True, help="User text to analyze.")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-dir", default=None, help="Milestone 2 trained model directory.")
    parser.add_argument("--model-type", choices=["BERT", "DistilBERT"], default="BERT")
    parser.add_argument("--history-csv", default=None, help="Emotion history CSV; enables trend tracking.")
    parser.add_argument("--feedback-csv", default=None, help="Feedback CSV; enables feedback learning.")
    parser.add_argument("--wellness-csv", default=None)
    parser.add_argument("--interactions-csv", default=None)
    parser.add_argument("--semantic-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--no-persist-history", action="store_true")
    args = parser.parse_args()

    profile = UserProfile(user_id=args.user_id)
    result = run_milestone3_part2(
        args.text,
        profile,
        emotion_model_dir=args.model_dir,
        model_type=args.model_type,
        wellness_content_csv=args.wellness_csv,
        interactions_csv=args.interactions_csv,
        emotion_history_csv=args.history_csv,
        feedback_csv=args.feedback_csv,
        top_k=args.top_k,
        semantic_model_name=args.semantic_model,
        persist_history=not args.no_persist_history,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("is_valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
