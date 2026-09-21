"""Simple user-user collaborative filtering over historical interactions."""
from __future__ import annotations
from collections import defaultdict
import math

class CollaborativeFilter:
    def __init__(self, interactions):
        self.by_user = defaultdict(dict)
        for x in interactions:
            self.by_user[x.user_id][x.content_id] = float(x.reward)

    @staticmethod
    def _cosine(a, b):
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def score(self, user_id, content_id):
        target = self.by_user.get(user_id, {})
        if not target: return 0.0
        weighted = denom = 0.0
        for other_id, other in self.by_user.items():
            if other_id == user_id: continue
            sim = self._cosine(target, other)
            if sim > 0 and content_id in other:
                weighted += sim * other[content_id]; denom += sim
        raw = weighted / denom if denom else 0.0
        return float(max(0.0, min(1.0, raw)))
