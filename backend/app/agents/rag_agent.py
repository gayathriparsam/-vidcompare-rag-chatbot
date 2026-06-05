"""LangGraph RAG agent with memory, streaming, and source citations.

Graph design (intentionally simple — fits a chat Q&A flow with citations):

  START -> retrieve -> respond -> END

`retrieve` queries Chroma for the latest user message, scoped to the current
session. It stuffs the formatted chunks into the `context` channel and records
`citations` so the UI can render "Source: Video A, chunk 3".

`respond` runs the LLM with: system prompt + context + full history, streaming
tokens. History persistence is handled by the MemorySaver checkpointer, keyed
by `thread_id = session_id`.
"""
from __future__ import annotations

import logging
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
        inputs = {
            "messages": [HumanMessage(content=user_message)],
            "video_a_meta": video_a_meta,
            "video_b_meta": video_b_meta,
            "session_id": session_id,
        }

        # Phase 1: run retrieve -> respond in non-streaming mode so we can
        # capture citations, then re-invoke the LLM streaming-only for tokens.
        # Simpler alternative: do retrieve separately here, then stream LLM.
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

        # Phase 2: stream LLM tokens. We use graph.stream(..., stream_mode="messages")
        # which yields (message_chunk, metadata) per token.
        sys_prompt = SYSTEM_PROMPT.format(
            video_a=_format_meta(video_a_meta),
            video_b=_format_meta(video_b_meta),
        )
        history_in_graph = [m for m in self.graph.get_state(config).values.get("messages", [])]
        full_messages: List = [SystemMessage(content=sys_prompt)]
        full_messages.extend(history_in_graph)
        full_messages.append(HumanMessage(content=user_message))
        if ctx_parts:
            full_messages.append(
                SystemMessage(content=f"RETRIEVED CONTEXT:\n\n" + "\n\n".join(ctx_parts))
            )

        llm = _build_llm()
        # Stream the LLM response token-by-token
        full_response_text = ""
        for chunk, _meta in llm.stream(full_messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if isinstance(token, list):
                # Some chat models return content as a list of parts
                token = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in token)
            if not token:
                continue
            full_response_text += token
            yield {"event": "token", "data": token}

        # Persist the exchange into the checkpointer for next turn
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
