"""Chunk transcripts, embed with OpenAI (or local fallback), store in ChromaDB.

Embeddings
----------
* Default: OpenAI `text-embedding-3-small` (1536 dims) when OPENAI_API_KEY is set
  and FORCE_LOCAL_EMBEDDINGS is false.
* Fallback: Chroma's built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2,
  384 dims, ONNX runtime). Zero API quota, ~80MB model cached on first use.

The two are NOT compatible dimensions, so the active backend is reflected in
the Chroma collection name (`..._local` vs default) to avoid mixing vectors.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings as LCEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import settings

logger = logging.getLogger(__name__)

_embeddings = None
_vector_store = None
_using_local = False


def _use_local() -> bool:
    return settings.force_local_embeddings or not settings.openai_api_key


class _LocalLCEmbeddings(LCEmbeddings):
    """LangChain-compatible wrapper around Chroma's DefaultEmbeddingFunction.

    Chroma's native embedding function only exposes __call__; LangChain's
    Chroma wrapper expects .embed_documents and .embed_query.
    """

    def __init__(self) -> None:
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ef([self._clean(t) for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._ef([self._clean(text)])[0]

    @staticmethod
    def _clean(t: str) -> str:
        return (t or "").replace("\n", " ").strip()


def get_embeddings():
    """Return the active embeddings object (OpenAI or local)."""
    global _embeddings, _using_local
    if _embeddings is None:
        if _use_local():
            logger.info("Using local Chroma default embedding (all-MiniLM-L6-v2).")
            _embeddings = _LocalLCEmbeddings()
            _using_local = True
        else:
            logger.info("Using OpenAI embeddings (%s).", settings.embedding_model)
            _embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )
            _using_local = False
    return _embeddings


def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        coll = settings.chroma_collection + ("_local" if _use_local() else "")
        _vector_store = Chroma(
            collection_name=coll,
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

    # Chroma expects a list-of-lists (or a callable) for embeddings; add_texts
    # will use the embedding_function from the constructor.
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
