# Backend (FastAPI + LangGraph RAG)

## Run

```bash
python -m pip install -r requirements.txt
cp .env.example .env   # fill OPENAI_API_KEY
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: <http://localhost:8000/docs>

## API

### `GET /healthz`
Returns service status, the configured LLM/embedding model, and whether
OpenAI + IG cookies are configured. Useful for the frontend to show a
green/amber dot.

### `POST /api/analyze`
```json
{ "url_a": "https://www.youtube.com/watch?v=...", "url_b": "https://www.instagram.com/reel/..." }
```
- Detects platform per URL.
- Fetches metadata + transcript (with Whisper fallback).
- Chunks transcripts, embeds with OpenAI, upserts into Chroma.
- Returns:
```json
{
  "session_id": "abc123...",
  "video_a": { "video_id":"A", "title": "...", "views": 1840000, "engagement_rate": 8.22, ... },
  "video_b": { ... },
  "chunks_indexed": 8
}
```

### `POST /api/chat` (Server-Sent Events)
```json
{ "session_id": "abc123...", "message": "Why did A get more engagement?" }
```
Streams:
```
event: citation
data: {"video_id":"A","chunk_index":3,"snippet":"..."}

event: token
data: "Vid"

event: token
data: "Compare"

...

event: done
data: {"citations":[...]}
```

### `GET /api/session/{session_id}`
Returns the metadata blob for both videos in a session.

## Environment variables

| Var               | Default                  | Notes                                                                |
|-------------------|--------------------------|----------------------------------------------------------------------|
| `OPENAI_API_KEY`  | _empty_                  | Required. Embeddings + chat + optional Whisper.                       |
| `LLM_MODEL`       | `gpt-4o-mini`            | Swap to `gpt-4o`, `claude-3-5-sonnet-latest`, etc.                   |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Swap to `text-embedding-3-large` for higher recall.                  |
| `CHROMA_PERSIST_DIR` | `./.chroma`           | Where Chroma persists vectors.                                       |
| `CHROMA_COLLECTION`  | `video_rag`            | Collection name.                                                     |
| `CHUNK_SIZE`      | `500`                    | Chars per chunk.                                                     |
| `CHUNK_OVERLAP`   | `80`                     | Chars overlap between adjacent chunks.                               |
| `IG_COOKIE_FILE`  | _empty_                  | Optional. Path to a Netscape `cookies.txt` for IG.                   |
| `MANUAL_A_JSON`   | _empty_                  | Path to `{meta,transcript}` JSON. Use `manual://video-a` as URL.     |
| `MANUAL_B_JSON`   | _empty_                  | Same for B.                                                          |

## Smoke test

```bash
python -m scripts.smoke
```

Runs without an API key (skips embed + chat), and fully with one. Prints
PASS/FAIL per subsystem.

## Project layout

```
app/
  main.py                # FastAPI app + routes
  config.py              # pydantic-settings, env loading
  schemas/models.py      # Pydantic request/response models
  services/
    fetchers.py          # yt-dlp + youtube-transcript-api + Whisper
    ingest.py            # chunk + embed + Chroma
  agents/rag_agent.py    # LangGraph RAG with streaming + memory
scripts/
  smoke.py               # end-to-end self-test
sample_data/
  video_a.json           # demo video (YouTube-style)
  video_b.json           # demo video (Instagram-style)
```

## Known limitations

- **`MemorySaver` is in-process.** For multi-replica deployments, swap
  to `langgraph-checkpoint-postgres.PostgresSaver`.
- **In-process session dict.** Replace with Redis for >1 replica.
- **No rate limiting on `/api/chat`.** Add `slowapi` or nginx
  `limit_req` for prod.
- **No retry on OpenAI 429.** Add `tenacity` in `RAGAgent.stream()`.
