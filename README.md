# Mood Mentor — AI Emotion Understanding & Personalized Wellness

## 1. What this system does, end to end

```
                              INPUT SOURCES
        ┌───────────┬────────────┬────────────┬────────────┐
        │  Chat text │  .txt file │  .csv file │ PDF / Word │  <- Milestone 
        └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘
              │            │            │            │
              ▼            ▼            ▼            ▼
        ┌─────────────────────────────────────────────────┐
        │  STAGE 1 — INGESTION            (Task 1)         │
        │  read raw text from any source, normalize into   │
        │  one record shape: {id, source, raw_text, ...}   │
        └───────────────────────┬───────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  STAGE 2 — VALIDATION           (Task 1)         │
        │  reject empty / whitespace / non-text / None     │
        │  everything else marked is_valid=True             │
        └───────────────────────┬───────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  STAGE 3 — PREPROCESSING        (Task 2)         │
        │  clean_text -> tokenize -> remove_stopwords       │
        │            -> lemmatize -> processed_text         │
        │  (produces TWO outputs — see src/preprocessing.py)│
        └───────┬─────────────────────────────┬─────────────┘
                │ cleaned_text                  │ processed_text
                ▼                               ▼
        ┌───────────────────┐         ┌─────────────────────────┐
        │ STAGE 4a — VADER   │         │ STAGE 4b — TRANSFORMER   │
        │ baseline sentiment │         │ emotion classifier       │
        │ (Task 3)           │         │ (Milestone 2, not built  │
        │ pos/neg/neu/compound│        │  yet — see roadmap below)│
        └─────────┬──────────┘         └────────────┬─────────────┘
                  │                                  │
                  └───────────────┬──────────────────┘
                                   ▼
        ┌─────────────────────────────────────────────────┐
        │  STAGE 5 — REPORT / ANALYTICS   (Task 4)         │
        │  per-row results + run summary (counts, avg      │
        │  compound, sample size) -> CSV / JSON             │
        └───────────────────────┬───────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  STAGE 6 — PIPELINE INTEGRATION (Task 5)         │
        │  wires 1-5 together, one call, error-isolated     │
        │  per record -> importable by an API / service     │
        └─────────────────────────────────────────────────┘
```

This repo implements **Stages 1–3, 4a, 5, 6** — that's the full scope of
**Milestone 1: Text Ingestion & Baseline Sentiment**. Stage 4b (transformer
emotion classification) and the recommendation layer are Milestone 2+, sketched
in the roadmap section so you know what this is building toward.

---

## 2. Where this fits your wellness platform

Your flagship project already runs Spring Boot + Kafka + Python ML + React +
PostgreSQL + Redis. This repo is the **Python ML service** in that stack — it's
built as plain importable modules (`src/pipeline.py`) specifically so you can
drop a thin FastAPI or Flask wrapper around `run_pipeline()` / `run_pipeline_multi()`
and call it from Spring Boot over REST, or have it consume raw-text events off
a Kafka topic and publish sentiment-report events back. Nothing in `src/` assumes
a particular web framework, which keeps that integration choice open.

---

## 3. Repository layout

```
mood_mentor/
├── requirements.txt
├── README.md                    <- this file
├── data/
│   ├── sample_corpus.csv        <- 12-row labeled sample (Task 4 validation set)
│   └── sample_journal.txt       <- 10-line sample chat/journal log
├── src/
│   ├── ingestion.py              Task 1 — raw text / .txt / .csv ingestion + validation
│   ├── preprocessing.py          Task 2 — clean, tokenize, stopwords, lemmatize
│   ├── sentiment.py              Task 3 — VADER wrapper, scoring, classification
│   ├── report.py                 Task 4 — report DataFrame + summary stats
│   └── pipeline.py               Task 5 — end-to-end integration
└── tests/
    ├── test_ingestion.py         16 tests
    ├── test_preprocessing.py     12 tests
    ├── test_sentiment.py         10 tests
    ├── test_report.py            7 tests
    └── test_pipeline_integration.py   8 tests
```

All 56 tests pass (`python3 -m pytest tests/ -v`).

---

## 4. How to run it

```bash
pip install -r requirements.txt

# one-time NLTK data (only needed once per machine)
python3 -c "import nltk; [nltk.download(p) for p in \
  ['punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger_eng']]"

# run each module standalone (each has a __main__ smoke test)
python3 src/ingestion.py
python3 src/preprocessing.py
python3 src/sentiment.py

# run the full Milestone 1 pipeline against the sample data
python3 -m src.pipeline

# run the test suite
python3 -m pytest tests/ -v
```

To use it in your own code:

```python
from src.pipeline import run_pipeline, run_pipeline_multi

# single source
df, summary = run_pipeline("raw_text", "I'm feeling really anxious about work today")

# multiple sources in one batch (mirrors a real "chat + uploaded file" request)
df, summary = run_pipeline_multi([
    ("raw_text", "Some chat message"),
    ("txt_file", "/path/to/journal.txt"),
    ("csv_file", "/path/to/mood_log.csv"),
])
```

---

## 5. Task-by-task: what "done" looks like (Milestone 1 checklist)

### Task 1 — Text Ingestion ✅
- `src/ingestion.py`: `ingest_raw_text`, `ingest_txt_file`, `ingest_csv_file`, `read_input_data`
- Validated: empty string, whitespace-only, `None`, non-string, punctuation-only,
  missing file, wrong extension, missing CSV column, empty CSV/txt file
- Run: `python3 -m pytest tests/test_ingestion.py -v`

### Task 2 — Preprocessing ✅
- `src/preprocessing.py`: `clean_text`, `tokenize`, `remove_stopwords`, `lemmatize`
- Two real bugs were caught and fixed while building this:
  1. Punctuation *runs* like `"..."` / `"???"` weren't being filtered (only single
     punctuation chars were) — fixed with a dedicated `_is_punctuation_only` check.
  2. Lemmatization without POS tags left `"running"` and `"runs"` un-collapsed
     (WordNet defaults everything to noun) — fixed by running `nltk.pos_tag` first.
- Run: `python3 -m pytest tests/test_preprocessing.py -v`

### Task 3 — VADER Sentiment ✅
- `src/sentiment.py`: wraps `vaderSentiment`'s `SentimentIntensityAnalyzer`
- Every score comes straight from `analyzer.polarity_scores()` — a test
  (`test_different_inputs_give_different_scores_not_hardcoded`) explicitly
  asserts three different inputs produce three different compound scores,
  to catch any accidental hardcoding.
- Run: `python3 -m pytest tests/test_sentiment.py -v`

### Task 4 — Initial Report ✅
- `src/report.py` + `data/sample_corpus.csv` (12 labeled rows, including 2
  deliberately invalid ones to prove the report handles bad rows)
- Run: `python3 -m src.pipeline` to print a full summary + see it write
  `data/milestone1_report.csv`

### Task 5 — Pipeline Integration ✅
- `src/pipeline.py`: `run_pipeline`, `run_pipeline_multi`
- Per-record error isolation — one bad row in a CSV upload doesn't kill the batch
- Run: `python3 -m pytest tests/test_pipeline_integration.py -v`

---

## 6. Tools used (Milestone 1) and why

| Purpose | Tool | Why this one |
|---|---|---|
| Ingestion / tabular data | `pandas` | industry standard for CSV handling + will carry you into feature engineering later |
| Tokenization, stopwords, lemmatization | `nltk` | mature, well-documented, POS-tag-aware lemmatizer |
| Baseline sentiment | `vaderSentiment` | lexicon+rule based, **no training data needed**, tuned for short/informal/emoji-heavy chat text — the fastest way to a working baseline |
| Testing | `pytest` | standard Python testing; `tmp_path` fixture makes file-upload tests trivial |

## 7. Tools for what comes next (Milestone 2+)

| Milestone | Goal | Suggested tools |
|---|---|---|
| M2 — Emotion classification | Go beyond pos/neg/neu to actual emotions (joy, anger, sadness, fear, anxiety, etc.) | HuggingFace `transformers` (e.g. a `distilbert`/`roberta` model fine-tuned on GoEmotions or a similar emotion dataset), `datasets`, `torch` |
| M2 — Sequence/context understanding | Understand mood *across* a conversation, not just per-message | Sentence embeddings (`sentence-transformers`), a lightweight sequence model (LSTM or a small transformer) over per-message emotion vectors |
| M3 — More input formats | PDF/Word ingestion mentioned in your longer note | `pypdf`/`pdfplumber` for PDF, `python-docx` for Word — both just become new functions in `ingestion.py` feeding the same normalized record shape |
| M3 — Analytics & recommendations | Personalized wellness report | `scikit-learn` for trend/clustering on emotion history, plus a rules or LLM-based recommendation layer on top |
| Serving | Expose this to Spring Boot / React | `FastAPI` (async, typed, easy Kafka/REST wrapper around `pipeline.py`) |

## 8. Known limitations of the Milestone 1 baseline (worth knowing before M2)

- VADER is a **lexicon** model — it has no real understanding of context or
  sarcasm, and individual words can push a genuinely neutral sentence off zero
  (e.g. "The sky is grey" scores slightly positive because "grey" isn't in
  VADER's dictionary as neutral). This is expected and is exactly the gap
  Milestone 2's transformer model is meant to close.
- WordNet's lemmatizer doesn't handle every irregular verb (e.g. "ran" stays
  "ran" instead of becoming "run") — fine for a baseline; a fuller morphological
  analyzer (spaCy) would close this gap if it starts affecting downstream results.
- Only `.txt` and `.csv` ingestion exist right now; PDF/Word support is a small,
  additive change to `ingestion.py` when you get to Milestone 3.
