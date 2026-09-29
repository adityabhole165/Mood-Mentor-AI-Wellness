# MoodMentor — Milestone 3 Complete Validation

This project implements Milestone 3 Tasks 1–10 while preserving the Milestone 1/2 pipeline.

## Milestone 3 flow

```text
Text Input
  ↓
Preprocessing
  ↓
VADER Sentiment
  ↓
BERT / DistilBERT
  ↓
Emotion + Confidence
  ↓
Emotion Intensity / Emotional State
  ↓
Historical Emotion Trend
  ↓
Semantic Wellness Matching
  ↓
Personalized Interaction Model
  + Feedback Model
  ↓
Hybrid Recommendation Engine
  ↓
Dynamic Ranking
  ↓
Dynamic Explanation
  ↓
Feedback Storage
  ↓
Future Ranking Updates
  ↓
Task 9 Controlled Evaluation
  ↓
Task 10 API + SQLite persistence
```

## Tasks 1–5

- Task 1: emotion intensity and emotional state
- Task 2: ML personalization using emotion, intensity, preference, interactions and history
- Task 3: hybrid rule/content/preference/collaborative/emotion/history recommendation
- Task 4: dynamic score and deduplication
- Task 5: semantic similarity between emotional state and wellness content

## Tasks 6–10

### Task 6
`src/emotion_history.py` stores historical states and computes frequency, intensity trend, polarity trend, repeated emotions and recent state. History affinity is a real ranking feature.

### Task 7
`src/recommendation_feedback.py` stores view/accept/reject/rating/preference/interaction events. After enough valid labelled records, the feedback Random Forest is blended with the interaction model at 35% feedback / 65% interaction.

### Task 8
`src/recommendation_explainability.py` creates reasons and evidence from the actual ranking components.

### Task 9
`data/m3_evaluation_cases.json` is a fixed six-case dataset. Every case has `relevant_ids` and `accepted_ids`. Baseline and advanced ML are evaluated with Precision@K, Recall@K, F1, NDCG, acceptance rate, diversity and response time.

### Task 10
`src/api.py` exposes `/health`, `/analyze`, `/feedback`, and `/stats`. `src/database.py` persists analysis, recommendations, feedback and emotion history in SQLite. If your university project has a separate existing API/database, replace this adapter with that existing implementation while keeping the same data contract.

## Windows setup

```powershell
.venv\\Scripts\\Activate.ps1
pip install -r requirements_m3_part2.txt
python scripts/setup_nltk.py
```

If you already have a working environment, install only missing packages.

## Tests

Compile:

```powershell
python -m compileall -q src app.py
```

Run all tests:

```powershell
python -m pytest -q
```

Collect tests:

```powershell
python -m pytest --collect-only -q
```

Run Milestone 3 focused tests:

```powershell
python -m pytest -q tests/test_milestone3.py tests/test_milestone3_part2.py tests/test_task10_regression.py tests/test_recommendation_feedback.py tests/test_recommendation_evaluation.py tests/test_task10_api_database.py
```

## Run Streamlit

```powershell
streamlit run app.py
```

## Run API

```powershell
uvicorn src.api:app --reload
```

Then test:

```text
GET  /health
POST /analyze
POST /feedback
GET  /stats
```

## Mentor demo checks

1. Change current emotion and show ranking changes.
2. Add repeated historical fear records and show history affinity changes ranking.
3. Submit at least 8 valid labelled feedback records and show feedback learning becomes fitted.
4. Open explanation evidence for a recommendation.
5. Run the controlled six-case baseline-vs-advanced evaluation.
6. Verify acceptance rate comes from `accepted_ids`, not live feedback.
7. Call `/analyze` and confirm rows are written to SQLite.
8. Call `/feedback` and confirm both CSV feedback and SQLite feedback persistence.
9. Run the complete pytest suite with no collection errors.
