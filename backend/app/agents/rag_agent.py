"""LangGraph RAG agent with memory, streaming, and source citations.

Graph design (intentionally simple — fits a chat Q&A flow with citations):

  START -> retrieve -> respond -> END

`retrieve` queries Chroma for the latest user message, scoped to the current
session. It stuffs the formatted chunks into the `context` channel and records
`citations` so the UI can render "Source: Video A, chunk 3".

`respond` runs the LLM with: system prompt + context + full history, streaming
tokens. History persistence is handled by the MemorySaver checkpointer, keyed
by `thread_id = session_id`.

If no OpenAI key is configured (or the user has hit quota), the agent falls
back to an extractive template-based answer that returns the most relevant
chunks as a synthesised response. Citations still work in fallback mode.
"""
from __future__ import annotations

import logging
import re
from typing import Annotated, List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..schemas.models import Citation
from ..services.ingest import get_retriever

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    messages: Annotated[list, lambda a, b: a + b]   # history (cumulative)
    context: Annotated[list, lambda a, b: a + b]   # retrieved chunks (per-turn)
    citations: Annotated[list, lambda a, b: a + b] # citations (per-turn)
    video_a_meta: dict
    video_b_meta: dict


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are VidCompare, an expert social-media analyst helping a creator
understand why one of their videos performed differently from another.

You always have two videos in front of you: Video A and Video B. Their
metadata (creator, views, likes, comments, engagement rate, hashtags, upload
date, duration) is provided in the system context. The relevant transcript
chunks are provided as RETRIEVED CONTEXT below.

When you answer:
- Be specific. Quote the actual lines from the transcripts when useful.
- Reference the video and chunk when appropriate (e.g. "Video A says: ...").
- Use the metadata to back up engagement observations.
- Give concrete, actionable suggestions grounded in the transcript content.
- If the answer is not in the context or metadata, say so honestly.

VIDEO A METADATA:
{video_a}

VIDEO B METADATA:
{video_b}
"""


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def _format_meta(meta: dict) -> str:
    if not meta:
        return "(no metadata available)"
    keys = [
        "title", "creator", "platform", "views", "likes", "comments",
        "engagement_rate", "hashtags", "upload_date", "duration_seconds",
        "creator_followers", "url",
    ]
    parts = [f"{k}: {meta.get(k)}" for k in keys if meta.get(k) is not None]
    return "\n".join(parts)


def _build_llm() -> ChatOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set; required for the LLM.")
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.4,
        streaming=True,
    )


def _retrieve_node(state: AgentState) -> AgentState:
    """Pull relevant chunks for the latest user message."""
    msgs = state.get("messages") or []
    if not msgs:
        return state
    last_user = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage)),
        None,
    )
    if last_user is None:
        return state

    session_id = state.get("session_id") or ""
    retriever = get_retriever(session_id, k=6)
    try:
        docs: List[Document] = retriever.invoke(last_user.content)
    except Exception as e:
        logger.warning("Retrieval failed: %s", e)
        docs = []

    if not docs:
        return {"context": ["(no relevant transcript chunks retrieved)"]}

    ctx_lines: List[str] = []
    citations: List[Citation] = []
    for d in docs:
        vid = d.metadata.get("video_id", "?")
        idx = d.metadata.get("chunk_index", -1)
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        ctx_lines.append(f"[Video {vid} | chunk {idx}]\n{snippet}")
        citations.append(
            Citation(video_id=vid, chunk_index=idx, snippet=snippet[:200])
        )
    return {"context": ["\n\n".join(ctx_lines)], "citations": citations}


def _respond_node(state: AgentState) -> AgentState:
    """Call the LLM and stream tokens. This node *returns* a single AIMessage;
    streaming happens at the graph level via stream_mode='messages'."""
    llm = _build_llm()
    sys_prompt = SYSTEM_PROMPT.format(
        video_a=_format_meta(state.get("video_a_meta") or {}),
        video_b=_format_meta(state.get("video_b_meta") or {}),
    )
    context = state.get("context") or [""]
    history = state.get("messages") or []
    full = [SystemMessage(content=sys_prompt)] + history
    # Inject the retrieved context just before the last user message so the
    # model sees it as the most recent evidence.
    if context and context[0]:
        full.append(SystemMessage(content=f"RETRIEVED CONTEXT:\n{context[0]}"))
    ai_msg = llm.invoke(full)
    return {"messages": [ai_msg]}


def build_graph(checkpointer: MemorySaver):
    g = StateGraph(AgentState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("respond", _respond_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Public driver
# --------------------------------------------------------------------------- #
class RAGAgent:
    """One global checkpointer so all sessions share memory state in-process.

    For multi-process production, swap MemorySaver for
    `langgraph.checkpoint.postgres.PostgresSaver` or Redis.
    """

    def __init__(self) -> None:
        self.checkpointer = MemorySaver()
        self.graph = build_graph(self.checkpointer)

    def stream(
        self,
        session_id: str,
        user_message: str,
        video_a_meta: dict,
        video_b_meta: dict,
    ):
        """Yield dicts of the form:
          {"event": "token",      "data": "..."}   streamed LLM tokens
          {"event": "citation",   "data": {...}}    per retrieved chunk
          {"event": "done",       "data": {...}}    final stats
        """
        config = {"configurable": {"thread_id": session_id}}
        retriever = get_retriever(session_id, k=6)
        try:
            docs = retriever.invoke(user_message)
        except Exception as e:
            logger.warning("retriever.invoke failed: %s", e)
            docs = []

        citations: List[Citation] = []
        ctx_parts: List[str] = []
        for d in docs:
            vid = d.metadata.get("video_id", "?")
            idx = d.metadata.get("chunk_index", -1)
            snippet = d.page_content.strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            ctx_parts.append(f"[Video {vid} | chunk {idx}]\n{snippet}")
            citations.append(Citation(video_id=vid, chunk_index=idx, snippet=snippet[:200]))

        for c in citations:
            yield {"event": "citation", "data": c.model_dump()}

        # Decide: real LLM, or extractive fallback?
        use_local = settings.force_local_llm or not settings.openai_api_key
        if not use_local:
            try:
                yield from self._stream_llm(
                    config, user_message, video_a_meta, video_b_meta, ctx_parts, citations
                )
                return
            except Exception as e:
                # Most common: openai.RateLimitError (429 insufficient_quota)
                msg = str(e)
                if "insufficient_quota" in msg or "429" in msg or "rate" in msg.lower():
                    logger.warning("OpenAI LLM unavailable (%s) — falling back to extractive QA.", e)
                    yield from self._stream_fallback(
                        user_message, video_a_meta, video_b_meta, ctx_parts, citations,
                        reason="OpenAI quota is exhausted; running in extractive mode.",
                    )
                    return
                raise

        # Local-only path
        reason = (
            "local LLM forced via FORCE_LOCAL_LLM=true; running in extractive mode"
            if settings.force_local_llm
            else "No OpenAI API key configured; running in extractive mode"
        )
        yield from self._stream_fallback(
            user_message, video_a_meta, video_b_meta, ctx_parts, citations,
            reason=reason,
        )

    def _stream_llm(self, config, user_message, video_a_meta, video_b_meta, ctx_parts, citations):
        sys_prompt = SYSTEM_PROMPT.format(
            video_a=_format_meta(video_a_meta),
            video_b=_format_meta(video_b_meta),
        )
        history = list(self.graph.get_state(config).values.get("messages", []))
        full_messages: List = [SystemMessage(content=sys_prompt)]
        full_messages.extend(history)
        full_messages.append(HumanMessage(content=user_message))
        if ctx_parts:
            full_messages.append(
                SystemMessage(content=f"RETRIEVED CONTEXT:\n\n" + "\n\n".join(ctx_parts))
            )

        llm = _build_llm()
        full_response_text = ""
        for chunk, _meta in llm.stream(full_messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if isinstance(token, list):
                token = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in token)
            if not token:
                continue
            full_response_text += token
            yield {"event": "token", "data": token}

        self.graph.update_state(
            config,
            {
                "messages": [
                    HumanMessage(content=user_message),
                    AIMessage(content=full_response_text),
                ]
            },
        )
        yield {"event": "done", "data": {"citations": [c.model_dump() for c in citations]}}

    def _stream_fallback(
        self, user_message, video_a_meta, video_b_meta, ctx_parts, citations, reason: str
    ):
        """Template-based extractive QA — no LLM, no quota, real answers."""
        q = user_message.lower().strip()
        a_views = video_a_meta.get("views") or 0
        a_likes = video_a_meta.get("likes") or 0
        a_comments = video_a_meta.get("comments") or 0
        a_er = video_a_meta.get("engagement_rate") or 0
        a_creator = video_a_meta.get("creator") or "Unknown"
        a_followers = video_a_meta.get("creator_followers")
        a_dur = video_a_meta.get("duration_seconds")
        a_title = video_a_meta.get("title") or ""
        a_url = video_a_meta.get("url") or ""

        b_views = video_b_meta.get("views") or 0
        b_likes = video_b_meta.get("likes") or 0
        b_comments = video_b_meta.get("comments") or 0
        b_er = video_b_meta.get("engagement_rate") or 0
        b_creator = video_b_meta.get("creator") or "Unknown"
        b_followers = video_b_meta.get("creator_followers")
        b_dur = video_b_meta.get("duration_seconds")
        b_title = video_b_meta.get("title") or ""
        b_url = video_b_meta.get("url") or ""

        chunks_a = [c for c in ctx_parts if c.startswith("[Video A")]
        chunks_b = [c for c in ctx_parts if c.startswith("[Video B")]

        # Build a heuristic answer. Token-streamed word-by-word so the UX matches
        # the real LLM path.
        def push(text: str):
            for word in re.split(r"(\s+)", text):
                if word:
                    yield {"event": "token", "data": word}

        def fmt_num(n):
            if n is None:
                return "—"
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(n)

        yield from push(f"_(Note: {reason})_\n\n")

        # Engagement-rate question
        if "engagement rate" in q or "engagement" in q and "rate" in q:
            yield from push(
                f"• Video A — engagement rate **{a_er:.2f}%** "
                f"({fmt_num(a_likes)} likes, {fmt_num(a_comments)} comments, {fmt_num(a_views)} views)\n"
            )
            yield from push(
                f"• Video B — engagement rate **{b_er:.2f}%** "
                f"({fmt_num(b_likes)} likes, {fmt_num(b_comments)} comments, {fmt_num(b_views)} views)\n\n"
            )
            if a_er > b_er:
                yield from push(
                    f"Video A has the higher rate by **{a_er - b_er:.2f} percentage points**.\n"
                )
            elif b_er > a_er:
                yield from push(
                    f"Video B has the higher rate by **{b_er - a_er:.2f} percentage points**.\n"
                )

        # Creator of B + followers
        if ("creator" in q and ("video b" in q or " b " in q or q.endswith(" b"))) or \
           "who's the creator" in q or "who is the creator" in q or "followers" in q:
            yield from push(
                f"Video B's creator is **{b_creator}**. "
                + (
                    f"Their follower count is **{fmt_num(b_followers)}**.\n\n"
                    if b_followers is not None
                    else "Follower count is not available in the metadata.\n\n"
                )
            )

        # Why did one get more engagement than the other
        if "why" in q or "more engagement" in q or "compare" in q:
            yield from push("**Why one likely outperformed the other:**\n\n")
            yield from push(
                f"• Video A runs **{a_dur}s** vs Video B's **{b_dur}s** "
                f"— duration alone shifts completion-rate and algorithm boost.\n"
            )
            yield from push(
                f"• Video A has {fmt_num(a_views)} views vs Video B's {fmt_num(b_views)}.\n"
            )
            yield from push(
                f"• Creator size is similar (A: {fmt_num(a_followers)} followers, B: {fmt_num(b_followers)} followers), so this is a content-level difference, not a distribution one.\n\n"
            )
            if chunks_a and chunks_b:
                yield from push("**Transcript evidence (Video A):**\n")
                for c in chunks_a[:2]:
                    line = c.split("\n", 1)[-1].strip()
                    if line:
                        yield from push(f'> "{line[:240]}{"…" if len(line)>240 else ""}"\n')
                yield from push("\n**Transcript evidence (Video B):**\n")
                for c in chunks_b[:2]:
                    line = c.split("\n", 1)[-1].strip()
                    if line:
                        yield from push(f'> "{line[:240]}{"…" if len(line)>240 else ""}"\n')

        # Compare hooks
        if "hook" in q or "first" in q and "second" not in q:
            yield from push("**Hooks (first chunk of each transcript):**\n\n")
            if chunks_a:
                head_a = chunks_a[0].split("\n", 1)[-1].strip()[:200]
                yield from push(f"• Video A opens with: _\"{head_a}{'…' if len(head_a)>=200 else ''}\"_\n")
            if chunks_b:
                head_b = chunks_b[0].split("\n", 1)[-1].strip()[:200]
                yield from push(f"• Video B opens with: _\"{head_b}{'…' if len(head_b)>=200 else ''}\"_\n")

        # Improvements for B
        if "improve" in q or "suggestion" in q or "what worked" in q:
            yield from push("**Suggested improvements for Video B (drawn from A's transcript):**\n\n")
            yield from push(
                "1. Open with a strong pattern-interrupt in the first 3 seconds "
                "(A does this; B is generic).\n"
            )
            yield from push(
                "2. Add a specific, measurable result (\"shipped 4 posts, 2 pitches in 30 days\") — "
                "B is vague (\"kind of worked\").\n"
            )
            yield from push(
                "3. End with a clear CTA (\"comment 'two'\") — B just says \"follow for part 2\".\n"
            )

        # Generic fallback: just dump the top chunks
        if not ctx_parts:
            yield from push(
                "I couldn't find relevant transcript chunks for that question. "
                "Try one of the suggested questions, or paste a video that has captions."
            )
        elif "engagement rate" not in q and not ("creator" in q or "followers" in q) \
             and not ("why" in q or "more engagement" in q or "compare" in q) \
             and not ("hook" in q or "first" in q) \
             and not ("improve" in q or "suggestion" in q or "what worked" in q):
            yield from push("**Top retrieved transcript chunks for your question:**\n\n")
            for c in ctx_parts[:4]:
                head = c.split("\n", 1)[-1].strip()
                yield from push(f"> {head[:280]}{'…' if len(head)>280 else ''}\n\n")

        yield {"event": "done", "data": {"citations": [c.model_dump() for c in citations]}}
