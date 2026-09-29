# MoodMentor UI

A React (Vite) front end for the MoodMentor emotion-analysis and wellness
recommendation API. This mirrors the flow of the project's `app.py` Streamlit
app (Milestones 1–4), rebuilt against the FastAPI backend instead of calling
the pipeline in-process.

## What's here

```
src/
├── api/client.js               All calls to the FastAPI backend, plus a
│                                BERT-vs-DistilBERT comparison helper
├── utils/csv.js                Dependency-free CSV parse/build/download
├── components/
│   ├── AnalyzeForm.jsx          Chat text / TXT file / CSV file input modes
│   ├── EmotionResult.jsx        Sentiment + full emotional-state read-out
│   ├── ModelComparison.jsx      BERT vs DistilBERT side by side + agreement
│   ├── RecommendationList.jsx   Ranked recommendation feed
│   ├── RecommendationCard.jsx   Reasons, all feedback actions, score breakdown
│   ├── CandidatesTable.jsx      Every scored candidate, pre-threshold
│   ├── BatchTable.jsx           CSV-mode summary table + per-row drill-down
│   ├── FeedbackLog.jsx          Session feedback log
│   ├── StatsPanel.jsx           GET /stats as count cards
│   └── HistoryTrends.jsx        History, trends, search/filter, export, delete
├── App.jsx                      Tabs: "Check in" and "Dashboard"
├── main.jsx
└── index.css                    Design tokens + all styling (no UI framework)
```

## How it maps to the Streamlit app's milestones

- **Milestone 1 (ingestion + VADER)** — the three input modes on the left:
  chat text, a `.txt` upload, or a `.csv` upload (pick which column holds the
  text, cap how many rows to run). CSV mode gets a summary table with
  positive/neutral/negative counts and average compound score, downloadable
  as CSV.
- **Milestone 2 (BERT vs DistilBERT)** — chat/TXT mode runs both models in
  parallel and shows them side by side with an agreement badge, exactly like
  the Streamlit expander. CSV batch mode runs the single model selected in
  the form (to keep a large batch fast) — pick BERT or DistilBERT from the
  dropdown before submitting.
- **Milestone 3 (recommendations, ranking, feedback, personalization)** —
  each recommendation shows its reasons, a "score breakdown" toggle (the same
  components the ranker weighs: rule, personalized ML, feedback ML, emotion
  relevance, semantic similarity, preference, collaborative, history), and
  the full action set (helpful / complete / accept / not for me / skip /
  hide / star rating / preferred tags), all posted to `/feedback`. Below
  that, "How the ranking was built" shows every scored candidate before the
  relevance cutoff, and a session feedback log lists everything you've sent.
- **Milestone 4 (dashboard)** — the Dashboard tab has live `/stats` counts,
  plus history/trends/search/export/delete. Search filters the loaded
  emotion/recommendation/feedback rows client-side. The extended endpoints
  (`/history/*`, `/trends`, `/reports`, `DELETE /users/{id}/data`) show "not
  available on this backend" gracefully if the connected API is the base
  Milestone 3 `api.py` rather than the fuller combined build — everything
  else works either way.

Not carried over (no matching HTTP endpoint exists): PDF report generation,
the recommendation stress test, and running the pytest suite from the UI —
those are dev-time Streamlit conveniences that talk to the filesystem/DB
directly rather than through the API.

## Backend endpoints used

Always present:
- `GET /health`
- `POST /analyze`
- `POST /feedback`
- `GET /stats`

Used opportunistically (see the Milestone 4 note above):
- `GET /history/emotions`
- `GET /history/recommendations`
- `GET /history/feedback`
- `GET /trends`
- `GET /reports`
- `DELETE /users/{user_id}/data`

## Run it

```bash
npm install
cp .env.example .env       # set VITE_API_BASE_URL if your API isn't on 127.0.0.1:8000
npm run dev
```

Open the printed local URL (usually http://localhost:5173). Make sure the
FastAPI backend is running first (`uvicorn src.api:app --reload` from the
backend project), or the check-in form will show a connection error.

## Build for deployment

```bash
npm run build
```

Outputs to `dist/`. Set `VITE_API_BASE_URL` to your deployed API's URL before
building, since Vite inlines env vars at build time.
