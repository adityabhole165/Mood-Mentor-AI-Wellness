# Setup Guide

## Requirements

- Python 3.10 or newer
- Node.js 18+ (only for the React front end)
- About 3 GB of disk for PyTorch, transformers and sentence-transformers
- Docker (optional)

## 1. Python environment

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_m4.txt
python scripts/setup_nltk.py
```

`requirements_m4.txt` is the complete environment. `requirements.txt`,
`requirements_m3_part2.txt` and `requirements_m4.txt` overlap; keep
`requirements_m4.txt` (or rename it to `requirements.txt`) and delete the rest.

Install the package so the `moodmentor` command exists:

```powershell
pip install -e .
moodmentor --help
```

To install only what the lightweight parts need, and add the model stack later:

```powershell
pip install -e .          # core: pandas, nltk, vader, sklearn, fastapi, streamlit, reportlab
pip install -e ".[ml]"    # torch, transformers, datasets, accelerate, sentence-transformers
pip install -e ".[dev]"   # pytest, pytest-cov
```

## 2. Model weights

Expected folders:

```
models/bert_emotion/          config.json, model weights, tokenizer files
models/distilbert_emotion/    config.json, model weights, tokenizer files
```

Options: train locally (see MODEL_CARD.md), copy the folders you already
trained, or point the app elsewhere with `MOOD_MENTOR_BERT_DIR` and
`MOOD_MENTOR_DISTIL_DIR`. The sentence-transformers model
(`all-MiniLM-L6-v2` by default) is downloaded from Hugging Face on first use,
so the first run needs internet access.

## 3. Environment variables

All are optional.

| Variable | Default | Purpose |
|---|---|---|
| `MOOD_MENTOR_DB` | `data/mood_mentor.db` | SQLite database |
| `MOOD_MENTOR_HISTORY` | `data/emotion_history.csv` | Emotion history log |
| `MOOD_MENTOR_FEEDBACK` | `data/recommendation_feedback.csv` | Feedback log |
| `MOOD_MENTOR_INTERACTIONS` | `data/interactions.csv` | Interaction log |
| `MOOD_MENTOR_BERT_DIR` | `models/bert_emotion` | BERT weights |
| `MOOD_MENTOR_DISTIL_DIR` | `models/distilbert_emotion` | DistilBERT weights |
| `MOOD_MENTOR_SEMANTIC_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `MOOD_MENTOR_CORS_ORIGINS` | localhost 5173 and 3000 | Comma-separated allowed browser origins |
| `MOOD_MENTOR_API_KEY` | unset (auth off) | Enables API-key auth once you add it (SECURITY.md) |

Front end: copy `frontend/.env.example` to `frontend/.env` and set
`VITE_API_BASE_URL` (default `http://127.0.0.1:8000`).

## 4. Run

```powershell
streamlit run app.py                 # dashboard
uvicorn src.api:app --reload         # API
cd frontend; npm install; npm run dev
```

The API loads the BERT model in a background thread at startup. `/health`
reports `model_warmup` as `pending`, `warming`, `ready` or `failed: ...`.

## 5. Verify a clean install

Run this in a brand-new folder and virtual environment:

```powershell
py -m venv .venv-check
.venv-check\Scripts\Activate.ps1
pip install .
moodmentor --help
moodmentor stress --iterations 10
```

## 6. Build the project zip without junk

Do not zip `.venv`, `node_modules`, `.vite`, `.pytest_cache`, `__pycache__`,
`*.egg-info`, `frontend/dist` or the stray `frontend/React src.zip`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `LookupError: punkt` / `wordnet` | `python scripts/setup_nltk.py` |
| `ModuleNotFoundError: torch` | `pip install -e ".[ml]"` |
| Emotion stage says model missing | Add weights under `models/` or set the `*_DIR` variables |
| React UI shows network/CORS errors | Set `MOOD_MENTOR_CORS_ORIGINS` to the UI's origin; confirm the API is running |
| First `/analyze` is slow | Model warm-up; wait for `model_warmup: ready` |
