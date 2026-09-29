"""Milestone 4 security/input validation helpers."""

from __future__ import annotations

import re
from pathlib import Path


USER_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

ALLOWED_EVENTS = {
    "view",
    "helpful",
    "complete",
    "accept",
    "reject",
    "skip",
    "hide",
    "rate",
}


def validate_user_id(user_id: str) -> str:
    user_id = str(user_id or "").strip()

    if not USER_ID_RE.fullmatch(user_id):
        raise ValueError(
            "Invalid user_id. Use 1-64 letters, numbers, '.', '_' or '-'."
        )

    return user_id


def validate_event(event: str) -> str:
    event = str(event or "").strip().lower()

    if event not in ALLOWED_EVENTS:
        raise ValueError(f"Unsupported feedback event: {event}")

    return event


def safe_path(path: str, allowed_root: str = "data") -> str:
    p = Path(path).resolve()
    root = Path(allowed_root).resolve()

    try:
        p.relative_to(root)
    except ValueError:
        raise ValueError("Path is outside the allowed data directory.")

    return str(p)


def sanitize_search_query(
    query: str,
    max_length: int = 100,
) -> str:
    return str(query or "")[:max_length].strip()