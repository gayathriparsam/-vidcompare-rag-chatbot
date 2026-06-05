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

    chroma_persist_dir: str = "./.chroma"
    chroma_collection: str = "video_rag"

    chunk_size: int = 500
    chunk_overlap: int = 80

    # Cost guard: refuse to embed if total transcript > this many chars
    max_transcript_chars: int = 60_000


settings = Settings()
