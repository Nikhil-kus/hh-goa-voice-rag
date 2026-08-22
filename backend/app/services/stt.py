"""
Sarvam STT service — saaras:v3 REST API.

Endpoint: POST https://api.sarvam.ai/speech-to-text
Model: saaras:v3 with mode=transcribe
Supports: 22 Indian languages + English, auto language detection

Handles:
  - timeout
  - 1 retry on 5xx
  - structured error reporting
"""
from __future__ import annotations

import asyncio
import io
from typing import Optional, Tuple

import httpx

from app.config import settings
from app.utils.logger import get_logger
from app.utils.timing import StageTimer

logger = get_logger(__name__)

MAX_RETRIES = 1
RETRY_ON_STATUS = {500, 502, 503, 504}


async def transcribe(
    audio_bytes: bytes,
    content_type: str = "audio/wav",
    language_code: Optional[str] = None,
    timer: Optional[StageTimer] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Transcribe audio using Sarvam saaras:v3.

    Args:
        audio_bytes:   raw audio bytes (WAV, MP3, WebM, etc.)
        content_type:  MIME type of the audio
        language_code: BCP-47 hint (e.g. "hi-IN"). None = auto-detect.
        timer:         optional StageTimer to record STT latency

    Returns:
        (transcript, language_code_detected, error_message)
        On success: (transcript_str, lang_code, None)
        On failure: (None, None, error_str)
    """
    if not settings.sarvam_api_key:
        return None, None, "SARVAM_API_KEY not configured"

    if not audio_bytes:
        return None, None, "Empty audio"

    # Determine file extension for Sarvam
    ext_map = {
        "audio/wav": "audio.wav",
        "audio/wave": "audio.wav",
        "audio/webm": "audio.webm",
        "audio/mp3": "audio.mp3",
        "audio/mpeg": "audio.mp3",
        "audio/ogg": "audio.ogg",
        "audio/opus": "audio.opus",
    }
    filename = ext_map.get(content_type.split(";")[0].strip().lower(), "audio.wav")

    headers = {
        "api-subscription-key": settings.sarvam_api_key,
    }

    last_error: str = "Unknown error"

    for attempt in range(MAX_RETRIES + 1):
        try:
            stage_name = "stt" if attempt == 0 else f"stt_retry_{attempt}"
            ctx = timer.stage(stage_name) if timer else _null_ctx()

            async with ctx:
                async with httpx.AsyncClient(timeout=settings.stt_timeout_ms / 1000) as client:
                    files = {"file": (filename, io.BytesIO(audio_bytes), content_type)}
                    data = {
                        "model": settings.sarvam_model,
                        "mode": "transcribe",
                        "with_timestamps": "false",
                    }
                    if language_code and language_code != "unknown":
                        data["language_code"] = language_code

                    resp = await client.post(
                        settings.sarvam_stt_url,
                        headers=headers,
                        files=files,
                        data=data,
                    )

            if resp.status_code == 200:
                body = resp.json()
                transcript = body.get("transcript", "").strip()
                detected_lang = body.get("language_code")
                if not transcript:
                    return None, None, "Empty transcript returned by STT"
                return transcript, detected_lang, None

            elif resp.status_code in RETRY_ON_STATUS and attempt < MAX_RETRIES:
                last_error = f"Sarvam STT {resp.status_code}: {resp.text[:200]}"
                logger.warning(
                    "STT retrying",
                    extra={"attempt": attempt + 1, "status": resp.status_code},
                )
                await asyncio.sleep(0.5)
                continue

            else:
                last_error = f"Sarvam STT error {resp.status_code}: {resp.text[:200]}"
                logger.error("STT failed", extra={"status": resp.status_code, "body": resp.text[:200]})
                return None, None, last_error

        except httpx.TimeoutException:
            last_error = f"Sarvam STT timeout after {settings.stt_timeout_ms}ms"
            logger.warning("STT timeout", extra={"attempt": attempt})
            if attempt < MAX_RETRIES:
                continue
            return None, None, last_error

        except Exception as e:
            last_error = f"STT unexpected error: {str(e)}"
            logger.error("STT exception", extra={"error": str(e)})
            return None, None, last_error

    return None, None, last_error


class _null_ctx:
    """No-op async context manager used when no timer is provided."""
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
