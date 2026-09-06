# Mood Mentor — Milestone 1 & 2

This project implements the supplied Milestone 1 and Milestone 2 requirements:
text ingestion, preprocessing, VADER sentiment, BERT/DistilBERT multi-label
emotion classification, confidence scoring, model evaluation, ISEAR validation,
and end-to-end integration.

## Main fixes from the failing run

1. `data/isear_subset_sample.csv` was missing. A test fixture is now included.
2. `Trainer(..., tokenizer=...)` was incompatible with the installed
   Transformers version. BERT and DistilBERT now detect whether the installed
   API expects `processing_class` or the older `tokenizer` argument.
3. Training/evaluation are now separated: `src/train_models.py` creates a
   deterministic held-out split rather than evaluating only on training rows.
4. ISEAR validation prefers `data/isear_subset.csv`; the bundled
   `isear_subset_sample.csv` is only a smoke-test fixture.
5. `src/fetch_isear.py` can create a real held-out ISEAR subset from the public
   Hugging Face dataset mirror.
6. Model prediction rejects empty/non-string input instead of producing a
   meaningless prediction.
7. Model evaluation compares BERT and DistilBERT using dynamically calculated
   subset accuracy, average per-label accuracy, macro precision, macro recall,
   and macro F1.
8. All existing Milestone 1 ingestion, preprocessing, sentiment, report, and
   integration tests are retained.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the NLTK resources used by preprocessing:

```powershell
python -c "import nltk; [nltk.download(x) for x in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

## Milestone 1

Run all Milestone 1 tests:

```powershell
python -m pytest tests/test_ingestion.py tests/test_preprocessing.py tests/test_sentiment.py tests/test_report.py tests/test_pipeline_integration.py -v
```

Run the sample pipeline:

```powershell
python -m src.pipeline
```

## Milestone 2

### 1–3: Train BERT and DistilBERT

```powershell
python -m src.train_models
```

This downloads the base Hugging Face checkpoints if they are not already cached,
fine-tunes both models on `data/emotion_train_sample.csv`, and saves:

- `models/bert_emotion/`
- `models/distilbert_emotion/`

The six labels are exactly:

`joy, sadness, anger, fear, surprise, disgust`

The training target is multi-label, so sigmoid/BCE-style outputs are used rather
than softmax.

### 4: Confidence scores

```powershell
python -m src.confidence
```

The production confidence values come directly from model sigmoid outputs. The
module rejects out-of-range or suspiciously identical score distributions.

### 5: Model evaluation

```powershell
python -m src.evaluation
```

The evaluation path uses a deterministic held-out split. Replace the tiny sample
CSV with a larger labeled corpus for meaningful performance reporting.

### 6: ISEAR benchmark

For the real benchmark subset:

```powershell
python -m src.fetch_isear --size 200 --seed 42
python -m src.isear_validation
```

ISEAR has seven original labels: joy, fear, anger, sadness, disgust, shame, and
guilt. This project's six-label taxonomy has no shame/guilt, and ISEAR has no
surprise label, so the validator evaluates only the five overlapping labels.
Rows with shame/guilt are excluded; surprise cannot be benchmarked against ISEAR.

The generated `data/isear_subset.csv` must remain evaluation-only and must not be
added to the model training CSV.

### 7: Full integration

After both models have been trained:

```powershell
python -m src.pipeline_v2
```

Flow:

`input -> preprocessing -> VADER -> transformer -> emotion scores -> confidence -> result`

### 8–10: Final validation

```powershell
python -m src.milestone2_validation
```

This first runs the automated test suite. If model artifacts exist, it also runs
held-out BERT/DistilBERT evaluation and ISEAR validation.

## Application

```powershell
streamlit run app.py
```

The application supports raw text, TXT, and CSV ingestion. After model training,
the transformer stage can be used for the final emotion analysis workflow.

## Important evaluation note

The repository's 45-row `emotion_train_sample.csv` is intentionally a small
development/smoke-test dataset. Passing tests proves the implementation is wired
correctly; it does **not** establish production-level model accuracy. For a
credible Milestone 2 performance claim, train/evaluate on a substantially larger,
properly separated labeled corpus and report the resulting dynamically computed
metrics.
