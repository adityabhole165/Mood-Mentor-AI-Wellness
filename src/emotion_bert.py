"""
emotion_bert.py
Milestone 2 - Task 1: BERT Model Integration

Fine-tunes bert-base-uncased for MULTI-LABEL emotion classification across
the 6 project-specified categories (see src/emotion_dataset.py):
    joy, sadness, anger, fear, surprise, disgust

Flow (matches Task 1 spec):
    load_pretrained_model()  -> Load Pre-trained BERT Model
    configure_tokenizer()    -> Configure BERT Tokenizer
    prepare_training_dataset() -> Prepare Training Dataset
    fine_tune_model()        -> Fine-tune BERT Model
    save_trained_model()     -> Save Trained Model
    predict()                -> Test Model with Sample Text

Multi-label (not multi-class): a sentence can be joy=1 AND fear=1 at once,
so this uses BCEWithLogitsLoss + per-label sigmoid, NOT softmax. Setting
problem_type="multi_label_classification" on the HF config switches the
loss automatically inside AutoModelForSequenceClassification -- you don't
have to write the loss function by hand.
"""

from __future__ import annotations
import os
import torch
import inspect
from dataclasses import dataclass, asdict
from typing import Dict, List

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from .emotion_dataset import EMOTIONS, EmotionDataset, load_emotion_dataset

MODEL_NAME = "bert-base-uncased"
DEFAULT_SAVE_DIR = os.path.join("models", "bert_emotion")
MAX_LENGTH = 128


@dataclass
class EmotionPrediction:
    text: str
    scores: Dict[str, float]        # every emotion -> confidence score (sigmoid output)
    primary_emotion: str
    primary_confidence: float
    triggered_emotions: List[str]   # emotions above threshold
    threshold: float

    def to_dict(self):
        return asdict(self)


def load_pretrained_model(num_labels: int = len(EMOTIONS)):
    """Step 1: Load Pre-trained BERT Model, configured for multi-label output."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )


def configure_tokenizer():
    """Step 2: Configure BERT Tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def prepare_training_dataset(csv_path: str, tokenizer, max_length: int = MAX_LENGTH):
    """Step 3: Prepare Training Dataset -- returns a torch Dataset ready for Trainer."""
    texts, labels = load_emotion_dataset(csv_path)
    return EmotionDataset(texts, labels, tokenizer, max_length=max_length)


class WeightedTrainer(Trainer):
    """Trainer that uses BCEWithLogitsLoss(pos_weight=...) instead of HF's
    default unweighted multi-label loss, so a class-imbalanced training set
    doesn't push the model toward always predicting the majority label(s)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    # We never resume training from a checkpoint, so the optimizer/scheduler/
    # RNG state don't need to exist on disk at all. Overriding these as no-ops
    # guarantees they're never written, regardless of what save_only_model
    # does or doesn't do in the installed transformers version -- this is
    # what was crashing on a mid-write disk/IO error.
    def _save_optimizer_and_scheduler(self, output_dir):
        pass

    def _save_scaler(self, output_dir):
        pass

    def _save_rng_state(self, output_dir):
        pass

def fine_tune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    output_dir: str = os.path.join("models", "bert_emotion_checkpoints"),
    epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    class_weights=None,
) -> Trainer:
    """Step 4: Fine-tune BERT Model on the prepared dataset."""
    raw_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "save_total_limit": 1,          # only keep the single best/most-recent checkpoint
        "save_only_model": True,        # skip optimizer/scheduler state -- we never resume training, so no need to save it (this is what was crashing)
        "logging_steps": 5,
        "load_best_model_at_end": eval_dataset is not None,
        "report_to": [],
    }
    # Transformers renamed evaluation_strategy -> eval_strategy.
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    raw_kwargs[strategy_key] = "epoch" if eval_dataset is not None else "no"

    # Only pass kwargs this installed TrainingArguments version actually
    # accepts -- an unsupported nice-to-have should be skipped, not crash
    # the whole training run.
    strategy_kwargs = {k: v for k, v in raw_kwargs.items() if k in ta_params}
    dropped = [k for k in raw_kwargs if k not in ta_params]
    if dropped:
        print(f"Note: TrainingArguments here doesn't support {dropped}; skipping.")
    args = TrainingArguments(**strategy_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "class_weights": class_weights,
    }
    # Transformers renamed Trainer(tokenizer=...) to Trainer(processing_class=...)
    # across releases. Support both so the project works with the installed version.
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)
    trainer.train()
    return trainer


def save_trained_model(model, tokenizer, save_dir: str = DEFAULT_SAVE_DIR) -> None:
    """Step 5: Save Trained Model (+ tokenizer, so it's self-contained for reload)."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def load_trained_model(save_dir: str = DEFAULT_SAVE_DIR):
    """Reload a previously fine-tuned model + tokenizer from disk."""
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    model.eval()
    return model, tokenizer


def predict(
    text: str,
    model,
    tokenizer,
    threshold: float = 0.5,
    max_length: int = MAX_LENGTH,
) -> EmotionPrediction:
    """
    Step 6: Test Model with Sample Text.

    Every value here comes from model(**inputs) at inference time -- nothing
    is hardcoded (this is exactly what Task 4 checks for).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prediction text must be a non-empty string")
    model.eval()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=max_length,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze(0).tolist()

    scores = {emotion: float(p) for emotion, p in zip(EMOTIONS, probs)}
    primary = max(scores, key=scores.get)
    triggered = [e for e, s in scores.items() if s >= threshold]

    return EmotionPrediction(
        text=text,
        scores=scores,
        primary_emotion=primary,
        primary_confidence=scores[primary],
        triggered_emotions=triggered,
        threshold=threshold,
    )


if __name__ == "__main__":
    # Smoke test: fine-tune briefly on the small sample set and predict.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(base_dir, "data", "emotion_train_sample.csv")

    tokenizer = configure_tokenizer()
    model = load_pretrained_model()
    train_ds = prepare_training_dataset(train_csv, tokenizer)

    fine_tune_model(model, tokenizer, train_ds, epochs=1, batch_size=4)
    save_trained_model(model, tokenizer)

    sample = "I am excited about the new opportunity but nervous about the outcome."
    result = predict(sample, model, tokenizer)
    print(result.to_dict())