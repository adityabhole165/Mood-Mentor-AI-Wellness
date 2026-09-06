
"""
emotion_distilbert.py
Milestone 2 - Task 2: DistilBERT Model Integration
 
Same multi-label fine-tuning flow as src/emotion_bert.py, swapped to
distilbert-base-uncased (6 transformer layers vs BERT's 12 -- roughly 40%
fewer parameters and noticeably faster inference, typically at a small cost
in accuracy). Kept as a near-duplicate of emotion_bert.py ON PURPOSE: Task 5
needs to evaluate both models under identical conditions, so any difference
in results should come from the model itself, not from divergent
preprocessing/training code between the two files.
"""
 
from __future__ import annotations
import os
import torch
import inspect
from typing import Dict
 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
 
from .emotion_dataset import EMOTIONS, EmotionDataset, load_emotion_dataset
from .emotion_bert import EmotionPrediction, MAX_LENGTH  # reuse the same result shape
 
MODEL_NAME = "distilbert-base-uncased"
DEFAULT_SAVE_DIR = os.path.join("models", "distilbert_emotion")
 
 
def load_pretrained_model(num_labels: int = len(EMOTIONS)):
    """Step 1: Load DistilBERT Model."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification",
    )
 
 
def configure_tokenizer():
    """Step 2: Configure Tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)
 
 
def prepare_training_dataset(csv_path: str, tokenizer, max_length: int = MAX_LENGTH):
    """Step 3: Prepare Training Data."""
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
 
 
def fine_tune_model(
    model, tokenizer, train_dataset, eval_dataset=None,
    output_dir: str = os.path.join("models", "distilbert_emotion_checkpoints"),
    epochs: int = 4, batch_size: int = 8, learning_rate: float = 2e-5,
    class_weights=None,
) -> Trainer:
    """Step 4: Fine-tune Model."""
    raw_kwargs = {
        "output_dir": output_dir, "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size, "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate, "warmup_ratio": 0.1, "weight_decay": 0.01,
        "save_strategy": "epoch",
        "save_total_limit": 1,          # only keep the single best/most-recent checkpoint
        "save_only_model": True,        # skip optimizer/scheduler state -- we never resume training, so no need to save it (this is what was crashing)
        "logging_steps": 5,
        "load_best_model_at_end": eval_dataset is not None, "report_to": [],
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    raw_kwargs[strategy_key] = "epoch" if eval_dataset is not None else "no"
 
    strategy_kwargs = {k: v for k, v in raw_kwargs.items() if k in ta_params}
    dropped = [k for k in raw_kwargs if k not in ta_params]
    if dropped:
        print(f"Note: TrainingArguments here doesn't support {dropped}; skipping.")
    args = TrainingArguments(**strategy_kwargs)
 
    trainer_kwargs = {
        "model": model, "args": args, "train_dataset": train_dataset,
        "eval_dataset": eval_dataset, "class_weights": class_weights,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedTrainer(**trainer_kwargs)
    trainer.train()
    return trainer
 
 
def save_trained_model(model, tokenizer, save_dir: str = DEFAULT_SAVE_DIR) -> None:
    """Step 5: Save Trained Model."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
 
 
def load_trained_model(save_dir: str = DEFAULT_SAVE_DIR):
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    model.eval()
    return model, tokenizer
 
 
def predict(text: str, model, tokenizer, threshold: float = 0.5, max_length: int = MAX_LENGTH) -> EmotionPrediction:
    """Step 6: Run Sample Predictions."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prediction text must be a non-empty string")
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze(0).tolist()
    scores = {emotion: float(p) for emotion, p in zip(EMOTIONS, probs)}
    primary = max(scores, key=scores.get)
    triggered = [e for e, s in scores.items() if s >= threshold]
    return EmotionPrediction(
        text=text, scores=scores, primary_emotion=primary,
        primary_confidence=scores[primary], triggered_emotions=triggered, threshold=threshold,
    )
 
 
def compare_predictions(text: str, bert_model, bert_tok, distil_model, distil_tok) -> Dict:
    """Step 7: Compare BERT and DistilBERT Results on the same input text."""
    from . import emotion_bert
    bert_result = emotion_bert.predict(text, bert_model, bert_tok)
    distil_result = predict(text, distil_model, distil_tok)
    return {
        "text": text,
        "bert": bert_result.to_dict(),
        "distilbert": distil_result.to_dict(),
        "agree_on_primary": bert_result.primary_emotion == distil_result.primary_emotion,
    }
 
 
if __name__ == "__main__":
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
 
