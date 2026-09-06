"""
train_models.py
Fine-tunes BERT and DistilBERT on the project's multi-label training CSV.
Uses a deterministic held-out split so evaluation does not score on training
data, and a dynamically-computed pos_weight so an imbalanced CSV can't make
the model collapse onto a single label (see emotion_dataset.compute_pos_weight).
"""
from __future__ import annotations
import os

from sklearn.model_selection import train_test_split

from .emotion_dataset import (
    load_emotion_dataset, EmotionDataset, EMOTIONS,
    label_distribution, compute_pos_weight,
)
from . import emotion_bert, emotion_distilbert

MIN_POSITIVES_PER_LABEL = 20  # below this, fine-tuning is unlikely to learn real signal


def train_one(model_module, train_texts, train_labels, eval_texts, eval_labels, model_dir, class_weights):
    tokenizer = model_module.configure_tokenizer()
    model = model_module.load_pretrained_model()
    train_ds = EmotionDataset(train_texts, train_labels, tokenizer)
    eval_ds = EmotionDataset(eval_texts, eval_labels, tokenizer)
    model_module.fine_tune_model(
        model, tokenizer, train_ds, eval_dataset=eval_ds,
        epochs=4, batch_size=8, output_dir=model_dir + "_checkpoints",
        class_weights=class_weights,
    )
    model_module.save_trained_model(model, tokenizer, model_dir)
    return model, tokenizer


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "emotion_train_sample.csv")
    texts, labels = load_emotion_dataset(csv_path)

    dist = label_distribution(labels)
    print(f"Loaded {len(texts)} rows. Per-label positive counts: {dist}")
    thin = [e for e, n in dist.items() if n < MIN_POSITIVES_PER_LABEL]
    if thin:
        print(
            f"WARNING: labels {thin} have fewer than {MIN_POSITIVES_PER_LABEL} "
            "positive examples. Fine-tuning on this little signal per label "
            "commonly produces a near-constant model. Run "
            "`python -m src.build_ekman_dataset` first to regenerate a larger, "
            "balanced training CSV."
        )

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    class_weights = compute_pos_weight(train_labels)
    print(f"Computed pos_weight (per {EMOTIONS}): {class_weights.tolist()}")

    os.makedirs(os.path.join(base, "models"), exist_ok=True)

    train_one(
        emotion_bert, train_texts, train_labels, eval_texts, eval_labels,
        os.path.join(base, "models", "bert_emotion"), class_weights,
    )
    train_one(
        emotion_distilbert, train_texts, train_labels, eval_texts, eval_labels,
        os.path.join(base, "models", "distilbert_emotion"), class_weights,
    )
    print("Both models trained and saved under models/.")


if __name__ == "__main__":
    main()