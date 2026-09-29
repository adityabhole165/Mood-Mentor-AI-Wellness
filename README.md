# MoodMentor

MoodMentor analyses free-text check-ins and returns an emotional state plus
personalised, explainable wellness recommendations.

- **Sentiment** – VADER baseline (compound / positive / negative / neutral)
- **Emotion** – fine-tuned BERT and DistilBERT, multi-label over six emotions:
  joy, sadness, anger, fear, surprise, disgust, each with a confidence score
- **Emotional state** – dominant emotion, intensity, polarity, severity, mixed states
- **Recommendations** – hybrid engine (rules, content-based, preferences,
  collaborative filtering, emotion similarity, history), semantic matching with
  sentence embeddings, dynamic ranking, per-item explanations, feedback learning
- **Insights** – daily / weekly / monthly emotion trends, history, search and
  filters, CSV and PDF reports, user-scoped data deletion

> Wellness recommendations are general information. They are not medical
> diagnosis or treatment.

## Ways to use it

| Interface | Command | Default URL |
|---|---|---|
| Streamlit dashboard | `streamlit run app.py` | http://localhost:8501 |
| REST API (FastAPI) | `uvicorn src.api:app --reload` | http://localhost:8000/docs |
| React dashboard | `cd frontend && npm install && npm run dev` | http://localhost:5173 |
| CLI | `moodmentor --help` | – |
| Docker | `docker compose up --build` | 8501 and 8000 |

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_m4.txt
python scripts/setup_nltk.py
pip install -e .
python -m pytest -q
streamlit run app.py
```

Linux / macOS: replace the activate line with `source .venv/bin/activate`.

The BERT and DistilBERT weights are **not** stored in the repository. Put them
in `models/bert_emotion/` and `models/distilbert_emotion/`, or train them
(see [docs/MODEL_CARD.md](docs/MODEL_CARD.md)). Without them, the M1 pipeline
(ingestion, preprocessing, VADER) and the deterministic recommendation
components still run; the emotion stages report that the model is missing.

## Project layout

```
app.py                  Streamlit dashboard (M1-M4 in one app)
src/
  ingestion.py, preprocessing.py, sentiment.py, report.py, pipeline.py       M1
  emotion_bert.py, emotion_distilbert.py, confidence.py, evaluation.py,
  isear_validation.py, train_models.py, pipeline_v2.py, pipeline_v3.py       M2
  emotional_state.py, hybrid_recommender.py, ranking.py, semantic_matching.py,
  personalized_recommender.py, collaborative_filtering.py, emotion_history.py,
  recommendation_feedback.py, recommendation_explainability.py,
  recommendation_evaluation.py, pipeline_v4.py, model_cache.py               M3
  m4_dashboard.py, reporting.py, security.py, stress_test.py, database.py,
  api.py, cli.py                                                             M4
frontend/               React (Vite) client for the FastAPI backend
tests/                  pytest suite (M1-M4)
data/                   sample corpora, ISEAR subset, local CSV/SQLite storage
models/                 trained weights (not committed)
docs/                   guides and references (below)
```

## Documentation

| Document | Contents |
|---|---|
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Installation, environment variables, clean-environment check |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Using the dashboard, React UI, CLI |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Every REST endpoint with examples |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Models, training, evaluation, limitations |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, compose, model weights, production notes |
| [docs/SECURITY.md](docs/SECURITY.md) | Validation, privacy, deletion, API-key authentication |
| [docs/HANDOVER_CHECKLIST.md](docs/HANDOVER_CHECKLIST.md) | Milestone status and final validation |

## Tests

```powershell
python -m compileall -q src app.py
python -m pytest -q
```

## Privacy

Data is stored locally under `data/`. Do not commit real journal text or
identifiers. Deletion removes a user's SQLite rows; see
[docs/SECURITY.md](docs/SECURITY.md) for what it does not remove.
