# Backend extras (optional)

The React app works without these -- word highlights show a friendly note and
accounts fall back to browser-only demo accounts. Add them for the full experience.

1. Copy `src/explain.py` and `src/auth.py` into your backend's `src/` folder.
2. In `src/api.py`, after `app = FastAPI(...)`, add:

```python
from .explain import router as explain_router
from .auth import router as auth_router
app.include_router(explain_router)
app.include_router(auth_router)
```

3. Set a real secret on your server (HF Space -> Settings -> Variables and secrets):
   `MOOD_MENTOR_AUTH_SECRET` = a long random string (e.g. `python -c "import secrets;print(secrets.token_hex(32))"`).

4. Optional, to actually *enforce* login on an existing route:

```python
from fastapi import Depends
from .auth import current_user

@app.post("/analyze")
def analyze(req: AnalyzeRequest, user: str = Depends(current_user)):
    if user != req.user_id:
        raise HTTPException(status_code=403, detail="user_id does not match your account")
    ...
```
   If you do this, guests (who have no token) will get 401 -- decide whether you want guest mode on the public site.
