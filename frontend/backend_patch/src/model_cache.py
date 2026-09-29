"""
Process-wide caches for the expensive objects pipeline_v4 used to rebuild
on every single /analyze call: the fine-tuned BERT/DistilBERT model+tokenizer
(loaded from disk) and the sentence-transformers SemanticMatcher (loaded +
re-fit on the wellness content).

Reloading a transformer model from disk and re-encoding the content library
on every request is the actual cause of request timeouts under load or on
CPU-only machines -- none of it changes between requests, so it only needs
to happen once per process (or once per distinct model_dir / content set).

Nothing here changes prediction behavior -- same model, same weights, same
.fit() -- it just stops repeating expensive, unchanged work.
"""
from __future__ import annotations
from typing import Dict, Tuple

from . import emotion_bert, emotion_distilbert
from .semantic_matching import SemanticMatcher

_emotion_models: Dict[Tuple[str, str], tuple] = {}
_matchers: Dict[Tuple[str, tuple], SemanticMatcher] = {}


def get_emotion_model(model_type: str, model_dir: str):
    """Load (model, tokenizer) once per (model_type, model_dir); reuse after."""
    key = (model_type.upper(), model_dir)
    cached = _emotion_models.get(key)
    if cached is not None:
        return cached
    if key[0] == "DISTILBERT":
        loaded = emotion_distilbert.load_trained_model(model_dir)
    elif key[0] == "BERT":
        loaded = emotion_bert.load_trained_model(model_dir)
    else:
        raise ValueError("model_type must be 'BERT' or 'DistilBERT'")
    _emotion_models[key] = loaded
    return loaded


def get_semantic_matcher(model_name: str, contents) -> SemanticMatcher:
    """Build + fit a SemanticMatcher once per (model_name, content set); reuse after.

    Keyed on the content ids so the matcher automatically refits itself if
    the wellness content library changes, without needing a manual reset.
    """
    signature = tuple(sorted(c.content_id for c in contents))
    key = (model_name, signature)
    cached = _matchers.get(key)
    if cached is not None:
        return cached
    matcher = SemanticMatcher(model_name)
    if hasattr(matcher, "fit"):
        matcher.fit(contents)
    _matchers[key] = matcher
    return matcher


def clear():
    """Drop all cached models -- useful after retraining/redeploying a model."""
    _emotion_models.clear()
    _matchers.clear()
