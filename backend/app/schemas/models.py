"""Public schemas for the API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    video_id: str = Field(..., description="A or B")
    platform: str = Field(..., description="youtube | instagram")
    url: str
    title: str
    creator: str
    creator_followers: Optional[int] = None
    views: int
    likes: int
    comments: int
    hashtags: List[str] = []
    upload_date: Optional[str] = None
    duration_seconds: Optional[int] = None
    thumbnail: Optional[str] = None
    transcript_chars: int = 0
    engagement_rate: float = 0.0


class AnalyzeRequest(BaseModel):
    url_a: str
    url_b: str


class AnalyzeResponse(BaseModel):
    session_id: str
    video_a: VideoMetadata
    video_b: VideoMetadata
    chunks_indexed: int


class Citation(BaseModel):
    video_id: str
    chunk_index: int
    snippet: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
