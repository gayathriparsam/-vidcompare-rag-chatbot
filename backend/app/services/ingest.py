"""Chunk transcripts, embed with OpenAI, store in ChromaDB with video_id tags.

We keep ONE persistent Chroma collection across all sessions, scoping
retrieval at query time by `where={"session_id": <sid>}` so two creators
never see each other's vectors.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from ..config import settings

logger = logging.getLogger(__name__)

_embeddings: OpenAIEmbeddings | None = None
_vector_store: Chroma | None = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set; required for embeddings.")
        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    return _embeddings


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        _vector_store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=get_embeddings(),
            persist_directory=settings.chroma_persist_dir,
            client_settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _vector_store


def split_into_chunks(text: str, video_id: str, session_id: str) -> List[Dict]:
    """Return list of {page_content, metadata} dicts ready for Chroma."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    docs = []
    for i, p in enumerate(pieces):
        docs.append(
            {
                "page_content": p,
                "metadata": {
                    "video_id": video_id,
                    "session_id": session_id,
                    "chunk_index": i,
                },
            }
        )
    return docs


def index_transcripts(
    session_id: str,
    transcript_a: str,
    transcript_b: str,
) -> int:
    """Embed both transcripts into Chroma tagged with session_id. Returns count."""
    docs: List[Dict] = []
    docs.extend(split_into_chunks(transcript_a, "A", session_id))
    docs.extend(split_into_chunks(transcript_b, "B", session_id))

    if not docs:
        return 0

    texts = [d["page_content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    ids = [f"{session_id}_{d['metadata']['video_id']}_{d['metadata']['chunk_index']}" for d in docs]

    # Delete any prior chunks for this session (idempotent re-index)
    try:
        get_vector_store().delete(where={"session_id": session_id})
    except Exception as e:  # pragma: no cover
        logger.info("Chroma delete (cold start ok): %s", e)

    get_vector_store().add_texts(texts=texts, metadatas=metadatas, ids=ids)
    try:
        get_vector_store().persist()
    except Exception:  # newer chroma auto-persists
        pass
    return len(docs)


def get_retriever(session_id: str, video_ids: List[str] | None = None, k: int = 6):
    """Return a LangChain retriever scoped to this session (and optional videos)."""
    where: Dict = {"session_id": session_id}
    if video_ids:
        where = {"$and": [{"session_id": session_id}, {"video_id": {"$in": video_ids}}]}

    return get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": where},
    )
