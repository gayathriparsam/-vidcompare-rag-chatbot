# VidCompare — Submission One-Pager

## Project description (for the form, ~200 words)

**VidCompare** is a full-stack Retrieval-Augmented Generation chatbot that lets a
creator paste one YouTube URL and one Instagram Reel URL and immediately get a
side-by-side "creator analyst" that answers questions like *"Why did Video A get
3× the engagement of Video B?"* or *"What's the most quotable line in either
video?"* — with streamed token-by-token responses that **cite the exact
transcript chunks** (video label, character range, matched phrase) used to
produce each answer.

**Stack:** Next.js 14 (React, TypeScript, Tailwind) frontend with side-by-side
video cards and a streaming chat panel · FastAPI backend · LangGraph (RAG
agent with `MemorySaver` for cross-turn conversation state) · ChromaDB
vector store · `yt-dlp` + `youtube-transcript-api` + `openai-whisper` for
ingestion (caption → auto-translated → Whisper fallback) ·
`gpt-4o-mini` for chat, `text-embedding-3-small` for embeddings (or local
`all-MiniLM-L6-v2` ONNX embeddings + extractive-QA template as a **zero-cost
fallback** when OpenAI quota is exhausted).

**Why this stack:** every component is a one-line swap from a hobby
deployment to a production one (Chroma → pgvector, MemorySaver → PostgresSaver,
gpt-4o-mini → gpt-4o). Cost analysis (in README) shows **~$320/month at
1,000 creators/day** at 0.5¢/request.

---

## What to highlight in the Loom (suggested flow, ~3 min)

1. **00:00–00:30** — Show the empty UI, paste the two URLs, click **Analyze**.
   Note: status pill ("Backend: ok · embeddings: local") and the
   `/api/diag/instagram` endpoint for transparency about IG blocking.
2. **00:30–01:00** — Side-by-side video cards appear with engagement rates.
   Note that Video A and Video B were ingested with different sources
   (live YouTube + manual override for IG) and the system handles both.
3. **01:00–02:00** — Click each of the four suggested questions. Show the
   token-by-token streaming and the **citations panel** (which chunks were
   retrieved, from which video, with character ranges).
4. **02:00–02:30** — Ask a follow-up: *"Give me one tip to make Video B
   better."* Show that memory carries context (the answer references the
   earlier comparison).
5. **02:30–03:00** — Show `python scripts/smoke.py` running — all 4 stages
   PASS in ~2 seconds, demonstrating the offline / zero-cost path works.

## Suggested questions (in the UI, click to ask)

1. "Why did Video A get more engagement than Video B?"
2. "What's the most quotable line from either video?"
3. "Give me one tip to make Video B perform better."
4. "What topics do both videos cover in common?"

## Submission checklist

- [x] GitHub repo: `gayathriparsam/-vidcompare-rag-chatbot`
- [x] README with quick start, architecture, cost analysis
- [x] 12+ commits telling a story (foundation → UI → robustness → zero-cost → IG hardening → auth)
- [x] End-to-end smoke test passes offline
- [x] Auth: signup/login/JWT (optional, doesn't break the demo path)
- [x] License (MIT)
- [ ] Loom demo (you record — script above)
- [ ] Submit form (description above)

## Architecture at a glance

```
+---------------------+      SSE      +-------------------------+      +----------------+
|  Next.js 14 UI      |  <--------->  |  FastAPI + LangGraph    |  ->  |  ChromaDB      |
|  (side-by-side      |      REST     |  - RAGAgent             |      |  (in-process)  |
|   cards + chat)     |               |  - SSE streamer         |      +----------------+
+---------------------+               |  - fetchers (yt-dlp +   |              |
                                      |    transcript-api +     |              v
                                      |    Whisper)             |      +----------------+
                                      +-------------------------+      |  OpenAI API    |
                                                                       |  gpt-4o-mini   |
                                                                       |  embeddings-3  |
                                                                       +----------------+
                                                                       (OPTIONAL: local
                                                                        ONNX fallback)
```

## Why I'd hire the author of this

- The two non-obvious things this gets right that most submissions miss:
  1. **Streamed citations, not just streamed text.** Every token chunk is
     accompanied by the `[{video, char_start, char_end, snippet}]` metadata
     so the user can verify the answer — not "trust me bro" AI.
  2. **A real offline path.** No OpenAI quota? The app still works end-to-end
     (local embeddings + extractive-QA template) — not a polite 503.
- Production thinking: every external dep has a fallback (caption →
  auto-translated caption → Whisper → extractive), the env-var story is
  documented, the IG blocking is surfaced (not silently swallowed), the
  auth flow is optional (doesn't break the demo path), and the cost model
  is on the README not the LinkedIn post.

## Auth (added late, but done right)

- **Backend** (`backend/app/auth.py`): SQLite users table (no extra DB to run),
  bcrypt password hashing (cost 12), PyJWT HS256 tokens, 7-day expiry.
- **Endpoints**: `POST /api/auth/signup`, `POST /api/auth/login`,
  `GET /api/auth/me`, `GET /api/me/sessions`.
- **Frontend** (`frontend/app/components/LoginPanel.tsx`): toggleable sign-in
  panel in the header, JWT stored in `localStorage`, sent as
  `Authorization: Bearer <token>` on `/api/analyze`.
- **Optional, not enforced**: the existing demo flow works without an account.
  When you sign in, your past sessions are saved so you can revisit them.
- **Why done this way**: didn't want to risk breaking the 11-commit
  foundation. Auth is additive; remove the LoginPanel and the whole app
  still works.
