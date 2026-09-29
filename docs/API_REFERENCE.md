
# MoodMentor API reference

Run locally:

```powershell
uvicorn src.api:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

## GET /health
Returns service/database status.

## POST /analyze

```json
{
  "text": "I feel anxious about tomorrow",
  "user_id": "demo_user",
  "top_k": 5,
  "model_type": "BERT",
  "preferred_types": [],
  "preferred_tags": [],
  "blocked_content_ids": [],
  "persist_history": true
}
```

## POST /feedback

```json
{
  "user_id": "demo_user",
  "content_id": "W001",
  "event": "accept",
  "rating": 5,
  "preference_changes": {"preferred_tags": ["breathing"]}
}
```

Allowed events: `view`, `helpful`, `complete`, `accept`, `reject`,
`skip`, `hide`, `rate`.

## GET /stats

Returns persisted row counts for analysis, recommendations, feedback and
emotion history.

Production deployments should add authentication/authorization, HTTPS,
rate limiting, structured logging, secrets management and a production
database. The local API is an educational integration boundary, not a
production security system.
