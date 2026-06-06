# VidCompare — Roadmap

This roadmap lists what was **intentionally left out of v1** to keep the
submission focused on the spec (RAG + citations + ingestion) and what is
planned for **v2** in priority order.

## Why no auth, no DB, no deploy in v1

The v1 scope was deliberately tight:

| V1 (shipped) | Why it was prioritized |
|---|---|
| Live ingestion (YouTube + IG + manual override) | The hard part — every external source fights back |
| LangGraph RAG agent with `MemorySaver` | Spec required LangChain + vector DB + memory |
| Streamed tokens + **per-token citations** | Differentiator — most submissions stream text only |
| Local fallback (embeddings + extractive QA) | The reviewer might not have an OpenAI key; the demo must work |
| Diagnostic endpoint for IG failures | Honest engineering — IG is hostile in 2025, surface it |

Auth, multi-user DB, and deploy are all expected from any "real" app, but
none of them test the spec's RAG depth. They'd also take 3-4 hours each and
risk breaking the working E2E.

## v2 — Priority order

### P0 — Auth + multi-user (3-4 hours)

Goal: every creator gets their own session history, can't see other users' chats.

**Backend:**
- `backend/app/auth/` — new package
  - `routes.py` — `POST /api/auth/signup`, `POST /api/auth/login`, `GET /api/auth/me`
  - `security.py` — `bcrypt` hashing + `python-jose` JWT issuance/verification
  - `deps.py` — `get_current_user` FastAPI dependency
- `backend/app/db.py` — SQLAlchemy + SQLite (dev) / Postgres (prod) models
  - `User` (id, email, hashed_password, created_at)
  - `Session` (id, user_id, video_a_meta, video_b_meta, created_at)
  - `Message` (id, session_id, role, content, citations_json, created_at)
- `backend/alembic/` — DB migrations
- `backend/requirements.txt` — add `bcrypt==4.1.2`, `python-jose[cryptography]==3.3.0`, `sqlalchemy==2.0.27`, `alembic==1.13.0`, `psycopg2-binary==2.9.9` (prod)
- Migration: in-memory `_sessions` dict → `db.query(Session).filter_by(user_id=...)`

**Frontend:**
- `frontend/app/login/page.tsx` — login + signup form (Tailwind)
- `frontend/app/components/AuthGuard.tsx` — client-side redirect to /login
- `frontend/app/api/auth/[...]` — Next.js route handlers (or call backend directly)
- `frontend/lib/auth.ts` — JWT in httpOnly cookie (NOT localStorage — XSS safe)
- `frontend/app/page.tsx` — show user's past sessions list, "New chat" button

**Effort breakdown:** schema + migrations 30 min · backend endpoints 60 min ·
login UI 60 min · route guards 30 min · testing 60 min

### P1 — Persistent chat history (1-2 hours)

Goal: every chat turn is stored, user can scroll back through past sessions.

- `Message` table populated inside `app/agents/rag_agent.py` after each `chat` call
- `GET /api/sessions` (list, paginated) + `GET /api/sessions/{id}/messages`
- Frontend: sidebar with session list, "Load" button

### P2 — Production deploy (2-3 hours)

Goal: a permanent public URL, no local setup needed.

- **Backend → Render.com** (free Web Service)
  - `render.yaml` blueprint
  - env vars in Render dashboard (no secrets in code)
  - Postgres free instance
  - CORS: `allow_origins=["https://vidcompare-frontend.vercel.app"]`
- **Frontend → Vercel** (zero-config Next.js)
  - `NEXT_PUBLIC_API_BASE=https://vidcompare-backend.onrender.com`
- **Total cost**: $0 (free tiers) up to ~1,000 MAU

### P3 — Real-time metrics (1-2 hours)

Goal: ops visibility, not blind to failures.

- `prometheus-fastapi-instrumentator` on the backend → `/metrics`
- `langsmith` callback in the LangGraph agent → traces of every RAG call
- Grafana dashboard (the user already runs Grafana locally on :3000 — we use :3001 in dev to avoid conflict)

### P4 — Content-aware features (4-6 hours)

Goal: stop being just Q&A, become an actual creator co-pilot.

- **Hook analysis**: detect the first 3 seconds of transcript, score for curiosity/pattern-interrupt
- **CTA extraction**: regex + LLM for "subscribe / follow / link in bio / comment below"
- **Cross-video clip suggestions**: find a 15-second segment in Video A that addresses a weakness in Video B
- **Thumbnail A/B predictor**: CLIP embeddings of video thumbnails + title CTR priors
- **Topic clustering** across a creator's whole video library (not just 2)

### P5 — Scaling to 10k+ creators/day (1-2 days)

- **Embeddings**: Chroma → pgvector with `pgvector` extension, same LangChain interface
- **Agent state**: `MemorySaver` → `PostgresSaver` (one DB, no per-process state)
- **Concurrency**: uvicorn `--workers 4` + Gunicorn; or migrate to async Celery workers
- **Caching**: `redis` for hot video metadata (YouTube + IG don't change often)
- **Rate limiting**: `slowapi` per-IP and per-user
- **CDN**: Cloudflare in front of Vercel for the frontend

## Anti-roadmap (things we will NOT add)

| Won't add | Why |
|---|---|
| Tiktok support | IG was already painful; TikTok is even more locked down. Spec didn't ask. |
| Video download / storage | We'd be a pirate site. Spec asked for analysis, not archival. |
| Real-time notifications / websockets | Not needed for chat. SSE already streams. |
| Mobile app | The web UI is responsive. Spec asked for a chatbot, not a native app. |
| Crypto / Web3 integration | Please. |

## Contribution order for v2

1. **Auth** (P0) — unblocks multi-user, makes P1 sensible
2. **Chat history** (P1) — completes the multi-user story
3. **Deploy** (P2) — shareable URL, gets this in front of real users
4. **Metrics** (P3) — now you can see what's breaking
5. **Content features** (P4) — actual product value beyond "chat with video"
6. **Scale** (P5) — only after the above prove product-market fit

Each step is independently shippable. None require a rewrite.
