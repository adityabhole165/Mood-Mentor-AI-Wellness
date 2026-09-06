# Mood Mentor

AI-based text ingestion, sentiment, and emotion analysis pipeline.
Built in two milestones:

- **Milestone 1** — text ingestion, preprocessing, and baseline sentiment (VADER)
- **Milestone 2** — deep emotion classification (fine-tuned BERT + DistilBERT), confidence
  scoring, evaluation metrics, and ISEAR benchmark validation

A Streamlit UI (`app.py`) wires both milestones together into one app.

---

## 1. Folder structure

```
Mood Mentor/
│
├── app.py                     # Streamlit UI — wraps Milestone 1 + Milestone 2 into one app
├── requirements.txt           # All dependencies (Milestone 1 + Milestone 2)
├── README.md                  # This file
│
├── src/                       # All pipeline code (imported as a package: `python -m src.xxx`)
│   ├── __init__.py
│   │
│   │   ── Milestone 1 ──
│   ├── ingestion.py            # Task 1 — raw text / .txt / .csv ingestion + validation
│   ├── preprocessing.py        # Task 2 — cleaning, tokenization, lemmatization
│   ├── sentiment.py            # Task 3 — VADER baseline sentiment scoring
│   ├── report.py               # Task 4 — builds the results DataFrame + summary dict
│   ├── pipeline.py             # Task 5 — wires ingestion → preprocessing → sentiment → report
│   │
│   │   ── Milestone 2 ──
│   ├── emotion_dataset.py      # Shared: 6 emotion categories, CSV loader, torch Dataset,
│   │                           #   class-imbalance helpers (label_distribution, compute_pos_weight)
│   ├── build_ekman_dataset.py  # Builds a real, balanced training CSV from GoEmotions (Ekman mapping)
│   ├── emotion_bert.py         # Task 1 — BERT: load, tokenize, fine-tune, save, predict
│   ├── emotion_distilbert.py   # Task 2 — DistilBERT: same flow + compare_predictions()
│   ├── train_models.py         # Fine-tunes both models on data/emotion_train_sample.csv
│   ├── confidence.py           # Task 4 — confidence-score extraction/validation on top of predict()
│   ├── evaluation.py           # Task 5 — accuracy / precision / recall / macro-F1, BERT vs DistilBERT
│   ├── fetch_isear.py          # Downloads a held-out ISEAR benchmark subset
│   ├── isear_validation.py     # Task 6 — validates the model against the ISEAR benchmark
│   ├── pipeline_v2.py          # Task 7 — Milestone 1 pipeline + emotion stage, in one call
│   └── milestone2_validation.py# Task 10 — runs the test suite + full Milestone 2 validation report
│
├── tests/                      # pytest suite (Task 8: edge cases — empty input, emojis, long/short text, etc.)
│
├── data/                       # Not committed with real data by default — generated/downloaded locally
│   ├── emotion_train_sample.csv  # Training data for emotion_bert.py / emotion_distilbert.py
│   │                               (build with `python -m src.build_ekman_dataset`)
│   ├── isear_subset.csv          # ISEAR benchmark subset (build with `python -m src.fetch_isear`)
│   ├── sample_journal.txt        # Small sample input for pipeline.py's __main__ demo
│   ├── sample_corpus.csv         # Small sample CSV input for pipeline.py's __main__ demo
│   └── milestone1_report.csv     # Output of pipeline.py's __main__ demo run
│
└── models/                     # Fine-tuned model artifacts — created by train_models.py, not committed
    ├── bert_emotion/             # Final saved BERT model + tokenizer (used by app.py)
    ├── bert_emotion_checkpoints/ # Intermediate Trainer checkpoints
    ├── distilbert_emotion/       # Final saved DistilBERT model + tokenizer (used by app.py)
    └── distilbert_emotion_checkpoints/
```

**Design rule the codebase follows throughout:** Milestone 2 code never modifies Milestone 1
files. `pipeline_v2.py` *adds* the emotion stage after VADER instead of editing `pipeline.py`,
so Milestone 1 stays independently testable and provably unaffected by Milestone 2 changes
(Task 7's requirement).

---

## 2. Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

`requirements.txt` is split into two parts: the Milestone 1 baseline (`pandas`, `numpy`, `nltk`,
`vaderSentiment`, `pytest`, `streamlit`) and Milestone 2's transformer stack (`transformers`,
`torch`, `datasets`, `scikit-learn`, `accelerate`) — install both together; Milestone 2 depends
on Milestone 1's output.

---

## 3. How to run

### Run the full test suite (Milestone 1 unit tests + Milestone 2 validation)

```bash
python -m src.milestone2_validation
```

This runs pytest across `tests/` first, then — if trained model artifacts already exist under
`models/` — also runs BERT/DistilBERT predictions, `evaluation.py`'s metrics, and
`isear_validation.py`'s benchmark check, printing one combined report.

### Build training data + train the emotion models (one-time, before first use)

```bash
python -m src.build_ekman_dataset --per_label 500   # writes data/emotion_train_sample.csv
python -m src.train_models                           # fine-tunes + saves BERT and DistilBERT
```

### Build the ISEAR benchmark subset (needed for Task 6)

```bash
python -m src.fetch_isear --size 200 --seed 42       # writes data/isear_subset.csv
```

### Run the Milestone 1 pipeline standalone (quick demo, no models needed)

```bash
python -m src.pipeline
```

### Run the full app (Milestone 1 + Milestone 2, interactive UI)

```bash
streamlit run app.py
```

In the sidebar you can:
- Choose input method: chat text, `.txt` file, or `.csv` file
- Toggle "Run BERT + DistilBERT emotion analysis" on/off
- Point at your trained model folders (`models/bert_emotion`, `models/distilbert_emotion`)
- Cap how many rows go through the transformer stage per batch (transformer inference is
  much slower than VADER, so this protects you from accidentally running 10,000 rows through
  BERT on a CPU)

CSV uploads need a text column — the pipeline defaults to a column named `text`.

---

## 4. Milestone 1 — Text Ingestion & Baseline Sentiment

**Goal:** get a clean, validated, end-to-end path from raw input to a baseline sentiment score,
with no ML training required — VADER is a lexicon/rule-based scorer, so this milestone works
correctly on day one.

| Task | What it validates | Implemented in |
|---|---|---|
| 1. Text Ingestion Workflow | Raw text entry, `.txt` upload, `.csv` upload all normalize into the same record shape; empty/invalid input is caught, not crashed on | `ingestion.py` |
| 2. Preprocessing Validation | Tokenization, stop-word removal, lemmatization, noise/punctuation handling, empty text, repeated spaces, varying lengths | `preprocessing.py` |
| 3. VADER Sentiment Validation | Positive/negative/neutral detection, compound/pos/neg/neu scores, all computed live (never hardcoded) | `sentiment.py` |
| 4. Initial Report Validation | Row-level results (input, processed text, sentiment) + summary stats (counts, averages, sample size) | `report.py` |
| 5. Complete Pipeline Integration | Every module's output correctly feeds the next; multiple input types can be run together in one batch | `pipeline.py` |

**Key implementation detail worth knowing:** `preprocessing.py` produces *two* separate text
fields, not one — `cleaned_text` (noise-only cleanup; case, punctuation, and emojis kept) and
`processed_text` (fully tokenized/lowercased/lemmatized, stopwords removed). VADER is scored on
`cleaned_text`, because VADER's accuracy actually depends on punctuation, capitalization, and
emojis ("GREAT!!!" scores differently than "great") — stripping those before VADER is a common
sentiment-pipeline bug that this design avoids on purpose.

**Output:** a DataFrame with one row per input record (`id`, `source`, `input_text`,
`processed_text`, `sentiment_label`, `compound`, `pos`, `neg`, `neu`, `is_valid`, `error`) plus a
summary dict — this is what Milestone 2 builds on top of.

---

## 5. Milestone 2 — Deep Emotion Classification & Validation

**Goal:** go beyond VADER's positive/negative/neutral into 6 specific emotions — **joy, sadness,
anger, fear, surprise, disgust** — using fine-tuned transformer models, with dynamically
computed confidence scores (never hardcoded) and real benchmark validation.

| Task | What it validates | Implemented in |
|---|---|---|
| 1. BERT Integration | Load `bert-base-uncased`, configure tokenizer, prepare dataset, fine-tune, save, predict on sample text | `emotion_bert.py` |
| 2. DistilBERT Integration | Same flow with `distilbert-base-uncased`; adds `compare_predictions()` to run both models side by side | `emotion_distilbert.py` |
| 3. Multi-Label Classification | A sentence can trigger more than one emotion at once (e.g. "excited but nervous" = joy **and** fear); implemented via `problem_type="multi_label_classification"` + sigmoid, not softmax | `emotion_dataset.py`, `emotion_bert.py` |
| 4. Confidence Score Validation | Per-emotion sigmoid score, primary emotion, which emotions cross the decision threshold — all read live from model output | `confidence.py` |
| 5. Model Evaluation | Accuracy (subset + per-label), precision, recall, macro-F1 for both models, computed via scikit-learn against a real held-out split | `evaluation.py` |
| 6. ISEAR Benchmark Validation | Validates predictions against the ISEAR emotion dataset (5 overlapping categories — see note below) | `isear_validation.py`, `fetch_isear.py` |
| 7. Milestone 1 + 2 Integration | Full pipeline: ingest → preprocess → VADER → BERT/DistilBERT → emotion classification → confidence scores → final result, in one call | `pipeline_v2.py` |
| 8. Edge Case Testing | Positive/negative/neutral/mixed text, short/long text, informal text, emojis, ambiguous statements, empty/invalid input | `tests/` |
| 9. Project Cleanup | No unused code/imports, no duplicated logic between BERT/DistilBERT modules beyond intentional near-duplication (Task 2 requires directly comparable results), no hardcoded predictions or scores | throughout `src/` |
| 10. Final Validation | Runs the entire Milestone 2 suite end to end and prints one combined report | `milestone2_validation.py` |

**ISEAR mismatch to know about (Task 6):** ISEAR's original labels are joy, fear, anger,
sadness, disgust, shame, guilt — no "surprise". So `isear_validation.py` only validates the 5
overlapping categories (joy, sadness, anger, fear, disgust); "shame"/"guilt" rows are excluded,
and "surprise" predictions simply can't be checked against ISEAR at all. This is a property of
the benchmark, not a bug in the validation code.

**Why training data quality matters here:** the classification head sitting on top of
BERT/DistilBERT starts randomly initialized — it only learns real signal if it sees enough
labeled examples per emotion during fine-tuning. `build_ekman_dataset.py` exists specifically to
replace a tiny hand-made CSV with thousands of real, balanced examples (via GoEmotions' Ekman
mapping), and `emotion_dataset.py`'s `compute_pos_weight()` further protects against any
remaining class imbalance by weighting the loss function — together these are what stand between
"the model actually learned emotions" and "the model just memorized whichever label was most
common."

**Output:** everything Milestone 1 produces, plus per-row `bert`/`distilbert` predictions (each
a dict of `primary_emotion`, `primary_confidence`, per-emotion `scores`, `triggered_emotions`,
and `threshold`), a `models_agree` flag, and — separately — evaluation metrics and an ISEAR
benchmark report for sign-off.