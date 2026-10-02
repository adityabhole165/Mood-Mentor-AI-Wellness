"""
explain.py -- word-level "Why this emotion?" for the React UI (POST /explain).

Method: occlusion. For each word, remove it, re-run the SAME emotion model and
measure how much the target emotion's probability drops. Words whose removal
hurts the most are the ones the model leaned on. No hardcoded outputs --
every weight comes from real forward passes.

Cost: one forward pass per word (capped by max_words, default 40), plus one for
the baseline. On CPU that is a couple of seconds for a typical entry.
"""
from __future__ import annotations

from typing import Callable, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class ExplainRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model_type: str = "BERT"
    target_emotion: str | None = None
    max_words: int = Field(default=40, ge=1, le=80)


def _load_predictor(model_type: str) -> Callable[[str], Dict[str, float]]:
    """Return text -> {emotion: probability} using the cached fine-tuned model."""
    from . import emotion_bert, emotion_distilbert
    from .model_cache import get_emotion_model

    kind = model_type.upper()
    if kind == "DISTILBERT":
        module, model_dir = emotion_distilbert, emotion_distilbert.DEFAULT_SAVE_DIR
    elif kind == "BERT":
        module, model_dir = emotion_bert, emotion_bert.DEFAULT_SAVE_DIR
    else:
        raise ValueError("model_type must be 'BERT' or 'DistilBERT'")
    from .preprocessing import preprocess_text

    model, tokenizer = get_emotion_model(kind, model_dir)

    def predict(text: str) -> Dict[str, float]:
        # Mirror the main pipeline: the model scores the *cleaned* text.
        cleaned = preprocess_text(text).cleaned_text or text
        return module.predict(cleaned, model, tokenizer).scores

    return predict


def explain_words(text: str, predict: Callable[[str], Dict[str, float]],
                  target: str | None = None, max_words: int = 40) -> dict:
    words = text.split()
    if not words:
        raise ValueError("Text has no words.")
    base = predict(text)
    if target is None:
        target = max(base, key=base.get)
    if target not in base:
        raise ValueError(f"Unknown emotion: {target}")
    base_score = float(base[target])

    deltas = []
    for i in range(len(words)):
        if i >= max_words:
            deltas.append(0.0)  # beyond the cap: not scored
            continue
        variant = " ".join(words[:i] + words[i + 1:]) or "."
        deltas.append(base_score - float(predict(variant)[target]))

    top = max((d for d in deltas if d > 0), default=0.0)
    out = []
    for word, delta in zip(words, deltas):
        weight = max(delta, 0.0) / top if top > 0 else 0.0
        out.append({"word": word, "weight": round(weight, 4), "delta": round(delta, 4)})
    return {"target": target, "base_score": round(base_score, 4),
            "scored_words": min(len(words), max_words), "words": out}


@router.post("/explain")
def explain(req: ExplainRequest):
    try:
        predict = _load_predictor(req.model_type)
        return explain_words(req.text, predict, req.target_emotion, req.max_words)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # model files missing, etc.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
