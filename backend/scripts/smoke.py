"""End-to-end smoke test.

Usage (no keys needed for fetcher tests):
    python -m scripts.smoke

With OPENAI_API_KEY set, also exercises the embedding + RAG agent paths.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.services.fetchers import fetch_video, fetch_manual
from app.services.ingest import index_transcripts, get_retriever


def header(t: str) -> None:
    print(f"\n=== {t} ===")


async def test_youtube_fetch() -> bool:
    header("YouTube live fetch")
    url = "https://www.youtube.com/watch?v=8jPQjjsBbIc"  # TED talk
    try:
        meta, tx = await fetch_video(url, "A")
    except Exception as e:
        print("  FAIL:", e)
        return False
    print(f"  title:   {meta.title[:60]}")
    print(f"  creator: {meta.creator}")
    print(f"  views:   {meta.views:,}  likes: {meta.likes:,}  comments: {meta.comments:,}")
    print(f"  engagement: {meta.engagement_rate}%")
    print(f"  transcript chars: {len(tx):,}")
    return bool(meta.title and tx)


def test_manual_fetch() -> bool:
    header("Manual override fetch")
    p = os.path.join(os.path.dirname(__file__), "..", "sample_data", "video_a.json")
    try:
        meta, tx = fetch_manual("A", p)
    except Exception as e:
        print("  FAIL:", e)
        return False
    print(f"  title:   {meta.title[:60]}")
    print(f"  views:   {meta.views:,}  engagement: {meta.engagement_rate}%")
    print(f"  transcript chars: {len(tx):,}")
    return bool(meta.title and tx)


async def test_ingest_and_retrieve() -> bool:
    header("Chroma ingest + retrieve")
    if not settings.openai_api_key:
        print("  SKIP (no OPENAI_API_KEY)")
        return True
    p_a = os.path.join(os.path.dirname(__file__), "..", "sample_data", "video_a.json")
    p_b = os.path.join(os.path.dirname(__file__), "..", "sample_data", "video_b.json")
    meta_a, tx_a = fetch_manual("A", p_a)
    meta_b, tx_b = fetch_manual("B", p_b)
    sid = uuid.uuid4().hex[:8]
    t0 = time.time()
    n = index_transcripts(sid, tx_a, tx_b)
    print(f"  indexed {n} chunks in {time.time()-t0:.2f}s")
    r = get_retriever(sid, k=4)
    docs = r.invoke("two minute rule productivity")
    print(f"  retrieved {len(docs)} docs for 'two minute rule productivity'")
    for d in docs:
        print(f"    -> Video {d.metadata['video_id']} chunk {d.metadata['chunk_index']}: "
              f"{d.page_content[:80]!r}")
    return n > 0 and len(docs) > 0


async def test_chat_stream() -> bool:
    header("LangGraph chat stream (1 turn)")
    if not settings.openai_api_key:
        print("  SKIP (no OPENAI_API_KEY)")
        return True
    p_a = os.path.join(os.path.dirname(__file__), "..", "sample_data", "video_a.json")
    p_b = os.path.join(os.path.dirname(__file__), "..", "sample_data", "video_b.json")
    meta_a, tx_a = fetch_manual("A", p_a)
    meta_b, tx_b = fetch_manual("B", p_b)
    sid = uuid.uuid4().hex[:8]
    index_transcripts(sid, tx_a, tx_b)

    from app.agents.rag_agent import RAGAgent
    agent = RAGAgent()
    print("  Q: Why did Video A get more engagement than Video B?")
    tokens = 0
    citations = 0
    t0 = time.time()
    for ev in agent.stream(
        sid, "Why did Video A get more engagement than Video B?",
        meta_a.model_dump(), meta_b.model_dump(),
    ):
        if ev["event"] == "token":
            tokens += 1
        elif ev["event"] == "citation":
            citations += 1
        elif ev["event"] == "done":
            break
    print(f"  streamed {tokens} token events, {citations} citations in {time.time()-t0:.2f}s")
    return tokens > 5 and citations > 0


async def main() -> int:
    results = []
    results.append(("yt fetch",      await test_youtube_fetch()))
    results.append(("manual fetch",  test_manual_fetch()))
    results.append(("ingest+retr",   await test_ingest_and_retrieve()))
    results.append(("chat stream",   await test_chat_stream()))

    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
