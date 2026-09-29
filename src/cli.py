
"""MoodMentor Milestone 4 command-line interface."""
from __future__ import annotations
import argparse
import json
import os


def main(argv=None):
    parser = argparse.ArgumentParser(prog="moodmentor", description="MoodMentor M1-M4 utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Run the existing analysis/recommendation API pipeline")
    a.add_argument("text")
    a.add_argument("--user-id", default="demo_user")
    a.add_argument("--model-type", choices=["BERT", "DistilBERT"], default="BERT")
    a.add_argument("--top-k", type=int, default=5)

    r = sub.add_parser("report", help="Generate a PDF report from stored data")
    r.add_argument("--user-id", default="demo_user")
    r.add_argument("--history", default="data/emotion_history.csv")
    r.add_argument("--db", default="data/mood_mentor.db")
    r.add_argument("--output", default="reports/mood_mentor_report.pdf")

    s = sub.add_parser("stress", help="Run recommendation performance smoke test")
    s.add_argument("--iterations", type=int, default=25)

    t = sub.add_parser("trends", help="Print emotion trend summary")
    t.add_argument("--user-id", default="demo_user")
    t.add_argument("--history", default="data/emotion_history.csv")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        from .pipeline_v4 import run_milestone3_part2
        from .recommendation_data import UserProfile
        result = run_milestone3_part2(args.text, UserProfile(args.user_id, [], [], set()),
                                       top_k=args.top_k, model_type=args.model_type)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "report":
        from .m4_dashboard import dashboard_snapshot
        from .reporting import generate_pdf_report
        snap = dashboard_snapshot(args.history, args.db, args.user_id)
        print(generate_pdf_report(snap, args.output))
    elif args.command == "stress":
        from .stress_test import run_recommendation_stress
        print(json.dumps(run_recommendation_stress(args.iterations), indent=2))
    elif args.command == "trends":
        from .m4_dashboard import dashboard_snapshot
        snap = dashboard_snapshot(args.history, os.getenv("MOOD_MENTOR_DB", "data/mood_mentor.db"), args.user_id)
        print(snap["daily_trends"].to_string(index=False))


if __name__ == "__main__":
    main()
