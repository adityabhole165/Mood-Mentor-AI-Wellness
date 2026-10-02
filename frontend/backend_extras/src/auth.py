"""
auth.py -- optional account system for the React UI.

  POST /auth/register {username, password} -> {username, token}
  POST /auth/login    {username, password} -> {username, token}
  GET  /auth/me       (Authorization: Bearer <token>) -> {username}

Standard library only: PBKDF2-SHA256 password hashes (salted), HMAC-signed
expiring tokens, a small in-memory login throttle. Users live in the same
SQLite file as the rest of the app (MOOD_MENTOR_DB).

Set MOOD_MENTOR_AUTH_SECRET to a long random string in production. If it is
missing a random one is generated at startup, which means every token is
invalidated whenever the server restarts.

To PROTECT an endpoint (optional), add:  user: str = Depends(current_user)
and check that `user == req.user_id`. The existing endpoints are left unchanged
so nothing that works today breaks.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/auth", tags=["auth"])

DB_PATH = os.getenv("MOOD_MENTOR_DB", os.path.join("data", "mood_mentor.db"))
SECRET = (os.getenv("MOOD_MENTOR_AUTH_SECRET") or secrets.token_hex(32)).encode()
TOKEN_TTL_SECONDS = int(os.getenv("MOOD_MENTOR_TOKEN_TTL", str(7 * 24 * 3600)))
USERNAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")  # same rule as security.validate_user_id
PBKDF2_ITERATIONS = 200_000
MAX_ATTEMPTS, WINDOW = 8, 60  # login attempts per username per minute

_attempts: dict[str, deque] = defaultdict(deque)


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=128)


def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "username TEXT PRIMARY KEY, salt TEXT NOT NULL, hash TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    return conn


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS).hex()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(username: str, now: float | None = None) -> str:
    payload = _b64(json.dumps({"u": username, "exp": int((now or time.time()) + TOKEN_TTL_SECONDS)}).encode())
    sig = _b64(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def verify_token(token: str) -> str | None:
    try:
        payload, sig = token.split(".", 1)
        expected = _b64(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(_unb64(payload))
        return data["u"] if data["exp"] >= time.time() else None
    except Exception:
        return None


def current_user(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    user = verify_token(authorization[7:].strip())
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def _throttle(key: str):
    now = time.time()
    q = _attempts[key]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many attempts. Try again in a minute.")
    q.append(now)


def _check_username(username: str) -> str:
    username = username.strip()
    if not USERNAME_RE.fullmatch(username):
        raise HTTPException(status_code=400, detail="Username must be 1-64 letters, numbers, '.', '_' or '-'.")
    return username


@router.post("/register")
def register(creds: Credentials):
    username = _check_username(creds.username)
    salt = secrets.token_bytes(16)
    try:
        with _conn() as conn:
            conn.execute("INSERT INTO users VALUES (?,?,?,?)",
                         (username, salt.hex(), _hash(creds.password, salt), time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username is already taken.")
    return {"username": username, "token": make_token(username)}


@router.post("/login")
def login(creds: Credentials):
    username = _check_username(creds.username)
    _throttle(username)
    with _conn() as conn:
        row = conn.execute("SELECT salt, hash FROM users WHERE username=?", (username,)).fetchone()
    # Always do the hash work so timing doesn't reveal whether the user exists.
    salt = bytes.fromhex(row[0]) if row else b"\x00" * 16
    candidate = _hash(creds.password, salt)
    if not row or not hmac.compare_digest(candidate, row[1]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return {"username": username, "token": make_token(username)}


@router.get("/me")
def me(user: str = Depends(current_user)):
    return {"username": user}
