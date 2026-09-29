
"""Milestone 4 report/export generation."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def build_report_frame(snapshot: dict) -> pd.DataFrame:
    rows = []
    for _, row in snapshot.get("history", pd.DataFrame()).iterrows():
        rows.append({
            "timestamp": row.get("timestamp"),
            "user_id": row.get("user_id"),
            "dominant_emotion": row.get("dominant_emotion"),
            "intensity": row.get("intensity"),
            "polarity": row.get("polarity"),
            "polarity_score": row.get("polarity_score"),
            "confidence": row.get("confidence"),
        })
    return pd.DataFrame(rows)


def build_csv_bundle(snapshot: dict) -> dict[str, bytes]:
    return {
        "emotion_history.csv": snapshot.get("history", pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        "daily_trends.csv": snapshot.get("daily_trends", pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        "recommendation_history.csv": snapshot.get("recommendations", pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        "feedback_history.csv": snapshot.get("feedback", pd.DataFrame()).to_csv(index=False).encode("utf-8"),
    }


def generate_pdf_report(snapshot: dict, output_path: str, title: str = "MoodMentor Emotional & Recommendation Report") -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    history = snapshot.get("history", pd.DataFrame())
    if history.empty:
        story.append(Paragraph("No emotion-history records matched the selected filters.", styles["BodyText"]))
    else:
        story.append(Paragraph(f"Emotion records: {len(history)}", styles["Heading2"]))
        cols = [c for c in ["timestamp", "dominant_emotion", "intensity", "polarity", "polarity_score"] if c in history.columns]
        data = [cols] + [[str(row.get(c, ""))[:45] for c in cols] for _, row in history.head(25).iterrows()]
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(table)
        story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"Recommendations recorded: {len(snapshot.get('recommendations', pd.DataFrame()))} · "
        f"Feedback events: {len(snapshot.get('feedback', pd.DataFrame()))}",
        styles["BodyText"],
    ))
    doc.build(story)
    return output_path
