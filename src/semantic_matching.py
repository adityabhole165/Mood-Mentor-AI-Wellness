"""Milestone 3 Task 5: transformer semantic wellness-content matching."""
from __future__ import annotations
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
from sklearn.feature_extraction.text import TfidfVectorizer

class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", allow_tfidf_fallback: bool = True):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name) if SentenceTransformer else None
        self.allow_tfidf_fallback = allow_tfidf_fallback
        self.vectorizer = None
        self.content_vectors = None
        self.content_ids = []

    @staticmethod
    def _cosine(a, b):
        a, b = np.asarray(a), np.asarray(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def fit(self, contents):
        self.content_ids = [c.content_id for c in contents]
        texts = [f"{c.title}. {c.description}. {' '.join(c.tags)} {' '.join(c.emotions)}" for c in contents]
        if self.model is not None:
            self.content_vectors = self.model.encode(texts, normalize_embeddings=True)
        elif self.allow_tfidf_fallback:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            self.content_vectors = self.vectorizer.fit_transform(texts)
        else:
            raise ImportError("sentence-transformers is required for semantic matching")
        return self

    def _encode(self, text):
        if self.model is not None:
            return self.model.encode([text], normalize_embeddings=True)[0]
        return self.vectorizer.transform([text])[0]

    def score(self, query, contents):
        if self.content_vectors is None or self.content_ids != [c.content_id for c in contents]:
            self.fit(contents)
        q = self._encode(query)
        result = {}
        for i, c in enumerate(contents):
            if self.model is not None:
                result[c.content_id] = self._cosine(q, self.content_vectors[i])
            else:
                result[c.content_id] = self._cosine(q.toarray()[0], self.content_vectors[i].toarray()[0])
        return result

    def match_emotional_state(self, emotional_state, original_text, contents):
        emotions = ", ".join(emotional_state.triggered_emotions) or emotional_state.dominant_emotion
        query = (f"Wellness need: {original_text}. Dominant emotion: {emotional_state.dominant_emotion}. "
                 f"Emotions: {emotions}. Intensity: {emotional_state.intensity:.3f}. "
                 f"Polarity: {emotional_state.polarity}. Priority: {emotional_state.severity}.")
        return self.score(query, contents)
