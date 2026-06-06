"""FastAPI app: /api/analyze, /api/chat (SSE), /healthz, /api/auth/*."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import auth
from .agents.rag_agent import RAGAgent
from .config import settings
from .schemas.models import AnalyzeRequest, AnalyzeResponse, ChatRequest
from .services.fetchers import fetch_manual, fetch_video
from .services.ingest import index_transcripts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("vidcompare")

app = FastAPI(title="VidCompare RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for a demo; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize the auth DB on startup. Idempotent.
@app.on_event("startup")
def _startup() -> None:
    auth.init_db()
    logger.info("auth DB initialised at %s", auth.DB_PATH)

# In-memory session store. Swap for Redis in prod.
_sessions: Dict[str, Tuple[dict, dict]] = {}
_agent = RAGAgent()


@app.get("/healthz")
def healthz():
    using_local = settings.force_local_embeddings or not settings.openai_api_key
    return {
        "status": "ok",
        "llm": settings.llm_model,
        "embedding": settings.embedding_model,
        "embedding_backend": "local" if using_local else "openai",
        "vector_db": "chroma",
        "openai_configured": bool(settings.openai_api_key),
        "openai_required": False,   # app works locally too
        "ig_cookie_configured": bool(settings.ig_cookie_file)
        and os.path.exists(settings.ig_cookie_file),
        "manual_a": bool(settings.manual_a_json),
        "manual_b": bool(settings.manual_b_json),
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    a, b = s
    return {"session_id": session_id, "video_a": a, "video_b": b}


# --- Auth routes ----------------------------------------------------------
class SignupBody(BaseModel):
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


@app.post("/api/auth/signup")
def signup(body: SignupBody):
    return auth.signup(body.email, body.password)


@app.post("/api/auth/login")
def login(body: LoginBody):
    return auth.login(body.email, body.password)


@app.get("/api/auth/me")
def me(email: str = Depends(auth.required_user)):
    return {"email": email}


# --- Authenticated session history (per user) ----------------------------
_user_sessions: Dict[str, list] = {}  # email -> [session_id, ...]


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, user_email: Optional[str] = Depends(auth.optional_user)):
    if not req.url_a or not req.url_b:
        raise HTTPException(400, "url_a and url_b are required")

    session_id = uuid.uuid4().hex[:16]
    try:
        if settings.manual_a_json and req.url_a.startswith("manual://"):
            meta_a, tx_a = fetch_manual("A", settings.manual_a_json)
        else:
            meta_a, tx_a = await fetch_video(req.url_a, "A")
        if settings.manual_b_json and req.url_b.startswith("manual://"):
            meta_b, tx_b = fetch_manual("B", settings.manual_b_json)
        else:
            meta_b, tx_b = await fetch_video(req.url_b, "B")
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch videos: {e}")

    chunks = 0
    try:
        chunks = index_transcripts(session_id, tx_a, tx_b)
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(500, f"Failed to index transcripts: {e}")

    _sessions[session_id] = (meta_a.model_dump(), meta_b.model_dump())
    if user_email:
        _user_sessions.setdefault(user_email, []).append(session_id)
    return AnalyzeResponse(
        session_id=session_id,
        video_a=meta_a,
        video_b=meta_b,
        chunks_indexed=chunks,
    )


@app.get("/api/me/sessions")
def my_sessions(email: str = Depends(auth.required_user)):
    """Returns the list of session_ids this user has created (newest first).
    Each session_id can be used with /api/session/{session_id} to retrieve the
    ingested video metadata."""
    sids = list(reversed(_user_sessions.get(email, [])))
    out = []
    for sid in sids[:20]:
        s = _sessions.get(sid)
        if s:
            a, b = s
            out.append({
                "session_id": sid,
                "video_a_title": (a.get("title") or "Video A")[:80],
                "video_b_title": (b.get("title") or "Video B")[:80],
            })
    return {"email": email, "count": len(out), "sessions": out}


@app.get("/api/diag/instagram")
async def diag_instagram(url: str = "https://www.instagram.com/reel/C3tK9bHJW1n/"):
    """Diagnose Instagram fetching: tries yt-dlp, reports exactly what's
    configured vs missing. Useful for the 'I pasted an IG URL and got an
    error' support path."""
    import yt_dlp

    report: Dict = {
        "url": url,
        "ig_browser": settings.ig_browser or None,
        "ig_cookie_file": settings.ig_cookie_file if settings.ig_cookie_file and os.path.exists(settings.ig_cookie_file) else None,
        "openai_configured": bool(settings.openai_api_key),
        "manual_b_configured": bool(settings.manual_b_json),
        "fetch": None,
        "fix": [],
    }

    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extractor_args": {"instagram": {"skip_warnings": True}},
    }
    if settings.ig_browser:
        opts["cookiesfrombrowser"] = (settings.ig_browser, None, None, None)
    elif settings.ig_cookie_file and os.path.exists(settings.ig_cookie_file):
        opts["cookiefile"] = settings.ig_cookie_file

    def _go() -> Dict:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False) or {}
        return {
            "title": info.get("title"),
            "uploader": info.get("uploader") or info.get("creator"),
            "view_count": info.get("view_count") or info.get("play_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
            "duration": info.get("duration"),
            "description": (info.get("description") or "")[:200],
        }

    try:
        report["fetch"] = await asyncio.to_thread(_go)
        report["status"] = "ok"
    except Exception as e:
        report["fetch"] = {"error": str(e)[:300]}
        report["status"] = "error"

    # Build actionable fix list
    if report["status"] == "error":
        err_text = (report["fetch"].get("error") or "")
        if not (settings.ig_browser or (settings.ig_cookie_file and os.path.exists(settings.ig_cookie_file))):
            report["fix"].append(
                "Instagram blocks anonymous GraphQL calls in 2025. Set ONE of these in backend/.env then restart uvicorn:\n"
                "  IG_BROWSER=chrome        # easiest: yt-dlp reads cookies from your logged-in Chrome\n"
                "  IG_COOKIE_FILE=./ig_cookies.txt   # export from a 'Get cookies.txt LOCALLY' extension"
            )
        if "Could not copy" in err_text and "cookie" in err_text.lower():
            report["fix"].append(
                f"yt-dlp cannot decrypt the cookie DB for {settings.ig_browser or 'your browser'} "
                "(common on Chromium 127+ with App-Bound Encryption). Workarounds:\n"
                "  1) Use IG_COOKIE_FILE: install a 'Get cookies.txt LOCALLY' extension while logged into instagram.com, export the cookies, save as backend/ig_cookies.txt, set IG_COOKIE_FILE=./ig_cookies.txt in .env, then restart.\n"
                "  2) OR use the manual override — set MANUAL_B_JSON=./sample_data/video_b.json and use 'manual://video-b' in the UI. This is the recommended demo path."
            )
        if not settings.openai_api_key:
            report["fix"].append(
                "Whisper fallback needs OPENAI_API_KEY (used to transcribe the audio if there are no captions)."
            )
        if not settings.manual_b_json:
            report["fix"].append(
                "OR use the manual override: set MANUAL_B_JSON=./sample_data/video_b.json and use 'manual://video-b' as the URL in the UI."
            )
    return report


@app.post("/api/chat")
async def chat(req: ChatRequest):
    s = _sessions.get(req.session_id)
    if not s:
        raise HTTPException(404, "session not found or expired. Re-run /api/analyze.")
    video_a, video_b = s

    async def event_gen():
        try:
            for ev in _agent.stream(req.session_id, req.message, video_a, video_b):
                yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], default=str)}\n\n"
        except Exception as e:
            logger.exception("chat stream failed")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
