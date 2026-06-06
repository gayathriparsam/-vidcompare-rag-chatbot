"""FastAPI app: /api/analyze, /api/chat (SSE), /healthz."""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Dict, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
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
    return AnalyzeResponse(
        session_id=session_id,
        video_a=meta_a,
        video_b=meta_b,
        chunks_indexed=chunks,
    )


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(404, "session not found")
    a, b = s
    return {"session_id": session_id, "video_a": a, "video_b": b}


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
