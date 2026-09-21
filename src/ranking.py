"""Milestone 3 Task 4: dynamic hybrid recommendation ranking."""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass
class RankedRecommendation:
    content_id: str
    title: str
    score: float
    rank: int
    reasons: list
    components: dict
    url: str = ""

    def to_dict(self): return asdict(self)

class RecommendationRanker:
    def __init__(self, low_relevance_threshold=0.20):
        self.low_relevance_threshold = low_relevance_threshold

    def _weights(self, intensity):
        # Higher intensity gives emotion/ML relevance slightly more influence.
        raw = {
            "ml": 0.20 + 0.08 * intensity,
            "emotion": 0.18 + 0.08 * intensity,
            "semantic": 0.15,
            "preference": 0.14,
            "cf": 0.10,
            "history": 0.08,
            "rule": 0.10,
            "novelty": 0.05,
        }
        total = sum(raw.values()); return {k: v / total for k, v in raw.items()}

    def score_row(self, row, state):
        w = self._weights(state.intensity)
        score = (w["ml"]*row["personalized_ml_score"] + w["emotion"]*row["emotion_relevance"] +
                 w["semantic"]*row["content_similarity"] + w["preference"]*row["preference_score"] +
                 w["cf"]*row["collaborative_score"] + w["history"]*row["history_affinity"] +
                 w["rule"]*row["rule_score"] + w["novelty"]*row["novelty"])
        return max(0.0, min(1.0, score))

    def rank(self, rows, state, top_k=5):
        unique = {}
        for row in rows:
            cid = row["content"].content_id
            if cid not in unique or self.score_row(row, state) > self.score_row(unique[cid], state): unique[cid] = row
        scored = []
        for row in unique.values():
            score = self.score_row(row, state)
            if score < self.low_relevance_threshold: continue
            reasons = []
            if row["emotion_relevance"] >= .35: reasons.append("emotion relevance")
            if row["content_similarity"] >= .35: reasons.append("semantic similarity")
            if row["preference_score"] >= .35: reasons.append("user preference")
            if row["collaborative_score"] >= .35: reasons.append("similar-user history")
            if row["history_affinity"] >= .35: reasons.append("previous positive interaction")
            scored.append((score, row, reasons or ["combined hybrid relevance"]))
        scored.sort(key=lambda x: (-x[0], x[1]["content"].content_id))
        results=[]
        weights=self._weights(state.intensity)
        for rank,(score,row,reasons) in enumerate(scored[:top_k],1):
            results.append(RankedRecommendation(row["content"].content_id,row["content"].title,round(score,6),rank,reasons,
                {"emotion_relevance":round(row["emotion_relevance"],6),"emotional_intensity":round(state.intensity,6),
                 "user_preference":round(row["preference_score"],6),"content_similarity":round(row["content_similarity"],6),
                 "collaborative_score":round(row["collaborative_score"],6),"previous_interaction":round(row["history_affinity"],6),
                 "rule_score":round(row["rule_score"],6),"personalized_ml_score":round(row["personalized_ml_score"],6),
                 "ranking_weights":weights}, row["content"].url))
        return results
