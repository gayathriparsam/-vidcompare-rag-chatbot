"""Application configuration loaded from environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # Force local embeddings (Chroma's built-in all-MiniLM-L6-v2) even if a
    # key is set. Useful when OpenAI quota is exhausted.
    force_local_embeddings: bool = False
    # Force the extractive (template-based) QA fallback even if a key is set.
    # The fallback returns the most relevant transcript chunks as a synthesised
    # answer — no LLM call, no quota needed.
    force_local_llm: bool = False

    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "video_rag"

    chunk_size: int = 500
    chunk_overlap: int = 80

    # Cost guard: refuse to embed if total transcript > this many chars
    max_transcript_chars: int = 60_000

    # Optional Instagram cookies (Netscape format) — see backend/README.md
    ig_cookie_file: str = ""
    # OR: auto-extract cookies from a logged-in browser (yt-dlp native).
    # Values: "chrome", "edge", "firefox", "brave", "opera", "chromium", "vivaldi", "safari"
    ig_browser: str = ""

    # Optional manual overrides (JSON). Useful for demo when IG blocks scraping.
    # Format: {"meta": {...}, "transcript": "..."}
    manual_a_json: str = ""
    manual_b_json: str = ""


settings = Settings()
