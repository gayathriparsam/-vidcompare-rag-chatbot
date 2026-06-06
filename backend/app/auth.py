"""Auth: SQLite users table, bcrypt password hashing, PyJWT tokens.

Design choices:
- SQLite (stdlib `sqlite3`) — no extra DB to run, no migrations.
- bcrypt with default cost factor (12) — slow by design, defeats brute force.
- JWT (HS256, 7-day expiry) — stateless, no session table to clean up.
- Optional auth: endpoints that take a token will *attribute* the session to
  a user, but the demo works fully without a token. This way the v1 demo
  path stays untouched.
- Email as the unique identifier (no username complexity).
- Passwords: min 8 chars, never logged, hashed before storage.
"""
from __future__ import annotations

import os
import sqlite3
import time
from typing import Optional

import bcrypt
import jwt
from fastapi import Header, HTTPException

from .config import settings

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "vidcompare_auth.db")
DB_PATH = os.path.abspath(DB_PATH)
JWT_ALG = "HS256"
JWT_TTL_SECONDS = 7 * 24 * 3600  # 7 days


def _conn() -> sqlite3.Connection:
    """Fresh connection per call (sqlite3 is fast to open, avoids thread issues)."""
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    """Create the users table if it doesn't exist. Idempotent."""
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                email        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password_hash BLOB  NOT NULL,
                created_at   INTEGER NOT NULL
            )
            """
        )
        c.commit()


def _validate_password(pw: str) -> None:
    if not isinstance(pw, str) or len(pw) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if len(pw) > 128:
        raise HTTPException(400, "password too long (max 128 chars)")


def _validate_email(email: str) -> None:
    if not isinstance(email, str) or "@" not in email or len(email) < 3 or len(email) > 254:
        raise HTTPException(400, "email looks invalid")
    # Cheap, non-strict check; full RFC 5321 is overkill here.
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain:
        raise HTTPException(400, "email looks invalid")


def signup(email: str, password: str) -> dict:
    """Create a user. Returns a JWT. Raises 409 if email already exists."""
    _validate_email(email)
    _validate_password(password)
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email.strip().lower(), pw_hash, int(time.time())),
            )
            c.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "email already registered")
    token = _make_token(email.strip().lower())
    return {"email": email.strip().lower(), "token": token, "created_at": int(time.time())}


def login(email: str, password: str) -> dict:
    """Verify credentials. Returns a JWT. Raises 401 on failure."""
    _validate_email(email)
    with _conn() as c:
        row = c.execute(
            "SELECT email, password_hash FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    if row is None or not bcrypt.checkpw(password.encode("utf-8"), bytes(row["password_hash"])):
        # Same error for both cases — never reveal whether the email exists.
        raise HTTPException(401, "invalid email or password")
    token = _make_token(row["email"])
    return {"email": row["email"], "token": token}


def _make_token(email: str) -> str:
    payload = {
        "sub": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_TTL_SECONDS,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALG)


def _decode_token(token: str) -> Optional[str]:
    """Returns the email if the token is valid, else None. Never raises."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALG])
        return payload.get("sub")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def optional_user(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """FastAPI dependency: returns the user's email if a valid Bearer token
    is present, else None. Use this for routes where auth is optional."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return _decode_token(parts[1].strip())


def required_user(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency: returns the user's email or raises 401.
    Use for endpoints that must be authenticated."""
    email = optional_user(authorization)
    if not email:
        raise HTTPException(401, "missing or invalid Authorization header")
    return email
