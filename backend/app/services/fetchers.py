"""Fetch metadata and transcripts for YouTube and Instagram Reels.

Strategy
--------
YouTube
  * Metadata: yt-dlp (no key)
  * Transcript: youtube-transcript-api first (free, instant) -> fall back to
    yt-dlp auto-captions -> fall back to OpenAI Whisper on downloaded audio.

Instagram
  * Metadata: yt-dlp (public reels, no key)
  * Transcript: yt-dlp auto-captions -> fall back to Whisper on downloaded audio.

Every method is wrapped in graceful failure with a clear error so the caller
can present a helpful message rather than a 500.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import yt_dlp
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from ..config import settings
from ..schemas.models import VideoMetadata

logger = logging.getLogger(__name__)


YT_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|embed/))([A-Za-z0-9_-]{11})"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def detect_platform(url: str) -> str:
    u = url.lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "instagram.com" in u:
        return "instagram"
    raise ValueError(f"Unsupported URL (need YouTube or Instagram): {url}")


def extract_yt_id(url: str) -> Optional[str]:
    m = YT_ID_RE.search(url)
    return m.group(1) if m else None


def compute_engagement(views: int, likes: int, comments: int) -> float:
    if views <= 0:
        return 0.0
    return round(((likes + comments) / views) * 100, 4)


def _run_ytdlp(opts: Dict[str, Any], url: str) -> Dict[str, Any]:
    """Run yt-dlp in a thread (it's blocking)."""
    def _go() -> Dict[str, Any]:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False) or {}

    return asyncio.to_thread(_go)


def _run_ytdlp_download(opts: Dict[str, Any], url: str) -> str:
    """Download and return the output filepath."""
    def _go() -> str:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # If a specific file was produced, return it
            if "requested_downloads" in info and info["requested_downloads"]:
                return info["requested_downloads"][0].get("filepath") or ydl.prepare_filename(info)
            return ydl.prepare_filename(info)

    return asyncio.to_thread(_go)


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #
def _yt_metadata_opts() -> Dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "noplaylist": True,
    }


async def fetch_youtube(url: str) -> Tuple[VideoMetadata, str]:
    """Return (metadata, transcript_text). Raises on hard failure."""
    info = await _run_ytdlp(_yt_metadata_opts(), url)
    if not info:
        raise RuntimeError("yt-dlp returned no info for YouTube URL")

    vid_id = info.get("id") or extract_yt_id(url) or "unknown"
    channel = info.get("uploader") or info.get("channel") or "Unknown"
    channel_followers = info.get("channel_follower_count")
    views = int(info.get("view_count") or 0)
    likes = int(info.get("like_count") or 0)
    comments = int(info.get("comment_count") or 0)
    upload = info.get("upload_date")  # YYYYMMDD
    duration = int(info.get("duration") or 0)
    title = info.get("title") or "(no title)"
    thumbnail = info.get("thumbnail")

    hashtags = []
    desc = info.get("description") or ""
    hashtags = sorted(set(re.findall(r"#(\w+)", desc)))

    transcript = await _youtube_transcript(vid_id, url)

    meta = VideoMetadata(
        video_id="",
        platform="youtube",
        url=url,
        title=title,
        creator=channel,
        creator_followers=channel_followers,
        views=views,
        likes=likes,
        comments=comments,
        hashtags=hashtags,
        upload_date=upload,
        duration_seconds=duration,
        thumbnail=thumbnail,
        transcript_chars=len(transcript),
        engagement_rate=compute_engagement(views, likes, comments),
    )
    return meta, transcript


async def _youtube_transcript(vid_id: str, url: str) -> str:
    """Try youtube-transcript-api -> yt-dlp auto-captions -> Whisper."""
    # 1. youtube-transcript-api
    try:
        ytt_api = YouTubeTranscriptApi()
        segments = ytt_api.fetch(vid_id)
        text = " ".join(seg.text.strip() for seg in segments)
        if text.strip():
            return text
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        logger.info("youtube-transcript-api miss for %s: %s", vid_id, e)
    except Exception as e:  # pragma: no cover
        logger.warning("youtube-transcript-api error: %s", e)

    # 2. yt-dlp auto-captions
    try:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en.*", "en"],
            "outtmpl": tempfile.mkdtemp() + "/%(id)s.%(ext)s",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles") or {}
            auto = info.get("automatic_captions") or {}
            for lang_set in (subs, auto):
                for lang, files in lang_set.items():
                    if lang.startswith("en") and files:
                        # pick the first vtt/srv1/etc
                        target = next(
                            (f for f in files if f.get("ext") in ("vtt", "srv1", "srv2", "ttml")),
                            files[0],
                        )
                        fmt_url = target.get("url")
                        if not fmt_url:
                            continue
                        import httpx
                        r = await asyncio.to_thread(lambda: httpx.get(fmt_url, timeout=30).text)
                        cleaned = _strip_vtt(r)
                        if cleaned.strip():
                            return cleaned
    except Exception as e:  # pragma: no cover
        logger.info("yt-dlp subtitle fetch failed: %s", e)

    # 3. Whisper fallback
    if settings.openai_api_key:
        try:
            return await _whisper_transcribe(url, "youtube")
        except Exception as e:
            logger.error("Whisper fallback failed for %s: %s", url, e)
    raise RuntimeError(
        "Could not obtain a transcript for this YouTube video. "
        "Auto-captions are disabled and no API key is configured for Whisper."
    )


# --------------------------------------------------------------------------- #
# Instagram
# --------------------------------------------------------------------------- #
def _ig_metadata_opts() -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extractor_args": {"instagram": {"skip_warnings": True}},
    }
    if settings.ig_cookie_file and os.path.exists(settings.ig_cookie_file):
        opts["cookiefile"] = settings.ig_cookie_file
    return opts


async def fetch_instagram(url: str) -> Tuple[VideoMetadata, str]:
    info = await _run_ytdlp(_ig_metadata_opts(), url)
    if not info:
        raise RuntimeError("yt-dlp returned no info for Instagram URL")

    creator = info.get("uploader") or info.get("creator") or info.get("channel") or "Unknown"
    creator_followers = info.get("follower_count") or info.get("uploader_follower_count")
    views = int(info.get("view_count") or info.get("play_count") or 0)
    likes = int(info.get("like_count") or 0)
    comments = int(info.get("comment_count") or 0)
    upload = info.get("upload_date")
    duration = int(info.get("duration") or 0)
    title = info.get("title") or info.get("description") or "(no title)"
    thumbnail = info.get("thumbnail")
    description = info.get("description") or title
    hashtags = sorted(set(re.findall(r"#(\w+)", description)))

    transcript = await _instagram_transcript(url, info)

    meta = VideoMetadata(
        video_id="",
        platform="instagram",
        url=url,
        title=title if len(title) < 200 else title[:200] + "…",
        creator=creator,
        creator_followers=creator_followers,
        views=views,
        likes=likes,
        comments=comments,
        hashtags=hashtags,
        upload_date=upload,
        duration_seconds=duration,
        thumbnail=thumbnail,
        transcript_chars=len(transcript),
        engagement_rate=compute_engagement(views, likes, comments),
    )
    return meta, transcript


async def _instagram_transcript(url: str, info: Dict[str, Any]) -> str:
    # 1. yt-dlp captions
    try:
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "outtmpl": tempfile.mkdtemp() + "/%(id)s.%(ext)s",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info2 = ydl.extract_info(url, download=False)
            for lang_set in (info2.get("subtitles") or {}, info2.get("automatic_captions") or {}):
                for lang, files in lang_set.items():
                    if files and (lang.startswith("en") or True):
                        target = next(
                            (f for f in files if f.get("ext") in ("vtt", "srv1", "srv2", "ttml")),
                            files[0],
                        )
                        fmt_url = target.get("url")
                        if not fmt_url:
                            continue
                        import httpx
                        r = await asyncio.to_thread(lambda: httpx.get(fmt_url, timeout=30).text)
                        cleaned = _strip_vtt(r)
                        if cleaned.strip():
                            return cleaned
    except Exception as e:  # pragma: no cover
        logger.info("IG caption fetch failed: %s", e)

    # 2. Whisper fallback
    if settings.openai_api_key:
        try:
            return await _whisper_transcribe(url, "instagram")
        except Exception as e:
            logger.error("Whisper fallback failed for IG %s: %s", url, e)
    raise RuntimeError(
        "Could not obtain a transcript for this Instagram reel. "
        "The reel has no captions and no API key is configured for Whisper."
    )


# --------------------------------------------------------------------------- #
# Whisper fallback (download audio then transcribe)
# --------------------------------------------------------------------------- #
async def _whisper_transcribe(url: str, platform: str) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot use Whisper fallback.")

    outdir = tempfile.mkdtemp(prefix="rag_audio_")
    outtmpl = os.path.join(outdir, "%(id)s.%(ext)s")
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
    }
    filepath = await _run_ytdlp_download(opts, url)
    # yt-dlp might have written the .mp3 next to the requested path
    mp3_path = os.path.splitext(filepath)[0] + ".mp3"
    if not os.path.exists(mp3_path):
        mp3_path = filepath  # may already be mp3

    client = OpenAI(api_key=settings.openai_api_key)

    def _go() -> str:
        with open(mp3_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text",
            )
        # gpt-4o transcribe returns string when format=text
        return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))

    text = await asyncio.to_thread(_go)
    return text


def _strip_vtt(vtt: str) -> str:
    """Strip VTT timestamps and tags, return plain text."""
    out: List[str] = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if "-->" in line:
            continue
        # strip inline tags like <c>...</c>
        line = re.sub(r"<[^>]+>", "", line)
        out.append(line)
    return " ".join(out).strip()


# --------------------------------------------------------------------------- #
# Public dispatch
# --------------------------------------------------------------------------- #
async def fetch_video(url: str, video_id: str) -> Tuple[VideoMetadata, str]:
    platform = detect_platform(url)
    if platform == "youtube":
        meta, transcript = await fetch_youtube(url)
    elif platform == "instagram":
        meta, transcript = await fetch_instagram(url)
    else:
        raise ValueError(f"Unsupported platform for {url}")
    meta.video_id = video_id
    # Guard: trim transcript if absurd
    if len(transcript) > settings.max_transcript_chars:
        transcript = transcript[: settings.max_transcript_chars]
        meta.transcript_chars = len(transcript)
    return meta, transcript


def fetch_manual(video_id: str, json_path: str) -> Tuple[VideoMetadata, str]:
    """Load a video from a local JSON file. Used when scraping is blocked."""
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta_dict = dict(data.get("meta") or {})
    meta_dict["video_id"] = video_id
    transcript = data.get("transcript") or ""
    # Recompute engagement rate for safety
    meta_dict["engagement_rate"] = compute_engagement(
        int(meta_dict.get("views") or 0),
        int(meta_dict.get("likes") or 0),
        int(meta_dict.get("comments") or 0),
    )
    meta_dict["transcript_chars"] = len(transcript)
    meta = VideoMetadata(**meta_dict)
    return meta, transcript
