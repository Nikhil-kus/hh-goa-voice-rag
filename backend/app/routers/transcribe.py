"""
POST /api/transcribe
Accepts multipart audio, calls Sarvam STT, returns transcript.
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import RefusalReason, TranscribeResponse
from app.services.guardrails import check_empty_transcript
from app.services.stt import transcribe
from app.utils.logger import get_logger
from app.utils.timing import StageTimer

logger = get_logger(__name__)
router = APIRouter()


@router.post("/api/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, WebM, etc.)"),
    language_code: str = Form(default="unknown",
                              description="BCP-47 language hint, or 'unknown' for auto-detect"),
):
    timer = StageTimer()

    # Validate audio
    with timer.stage("audio_validation"):
        audio_bytes = await file.read()
        ok, msg = check_empty_transcript(str(audio_bytes[:10]))  # quick empty check
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        if len(audio_bytes) < 1000:  # < 1KB = probably not real audio
            raise HTTPException(status_code=400, detail="Audio file too small")

    content_type = file.content_type or "audio/wav"
    lang_hint = None if language_code == "unknown" else language_code

    # STT
    transcript, detected_lang, error = await transcribe(
        audio_bytes=audio_bytes,
        content_type=content_type,
        language_code=lang_hint,
        timer=timer,
    )

    if error:
        logger.warning("Transcription failed", extra={"error": error})
        raise HTTPException(status_code=422, detail=error)

    # Validate transcript
    with timer.stage("transcript_validation"):
        ok, msg = check_empty_transcript(transcript or "")
        if not ok:
            raise HTTPException(status_code=422, detail=msg)

    latency = timer.all_ms()
    logger.info(
        "Transcription complete",
        extra={"lang": detected_lang, "transcript_len": len(transcript), "latency": latency},
    )

    return TranscribeResponse(
        transcript=transcript,
        language_code=detected_lang or "unknown",
        latency_ms=latency,
    )
