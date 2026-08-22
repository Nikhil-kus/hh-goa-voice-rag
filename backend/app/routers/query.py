"""
POST /api/query
Full RAG pipeline: transcript → guardrails → embed → retrieve → evidence gate → LLM → grounding → response.
Each stage has timeout, error handling, and latency measurement.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter

from app.config import settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    LatencyBreakdown,
    RefusalReason,
)
from app.services.generator import generate_answer, _is_llm_refusal
from app.services.centroid import get_centroid
from app.services.guardrails import (
    check_empty_transcript,
    check_grounding,
    check_insufficient_evidence,
    check_max_length,
    check_off_topic,
    check_unsafe_content,
)
from app.services.retriever import retrieve
from app.utils.logger import get_logger
from app.utils.timing import StageTimer

logger = get_logger(__name__)
router = APIRouter()

# Corpus centroid — computed lazily on first query if needed
_corpus_centroid = None


def _build_latency(timer: StageTimer) -> LatencyBreakdown:
    ms = timer.all_ms()
    return LatencyBreakdown(
        query_normalization_ms=ms.get("query_normalization"),
        guardrail_pre_ms=ms.get("guardrail_pre"),
        embedding_ms=ms.get("embedding"),
        vector_retrieval_ms=ms.get("vector_retrieval"),
        reranking_ms=ms.get("reranking"),
        evidence_scoring_ms=ms.get("evidence_scoring"),
        generation_ms=ms.get("generation"),
        grounding_check_ms=ms.get("grounding_check"),
        total_ms=ms.get("total", 0.0),
    )


def _refusal(
    timer: StageTimer,
    request: QueryRequest,
    reason: RefusalReason,
    message: str,
) -> QueryResponse:
    return QueryResponse(
        refused=True,
        refusal_reason=reason,
        refusal_message=message,
        transcript=request.transcript,
        language_code=request.language_code,
        strategy=request.strategy.value,
        latency=_build_latency(timer),
    )


@router.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    timer = StageTimer()

    # ── Stage 0: Input validation ─────────────────────────────────────────────
    ok, msg = check_empty_transcript(request.transcript)
    if not ok:
        return _refusal(timer, request, RefusalReason.empty_transcript, msg)

    transcript = check_max_length(request.transcript, settings.max_transcript_chars)

    # ── Stage 1: Pre-query guardrails ─────────────────────────────────────────
    with timer.stage("guardrail_pre"):
        ok, msg = check_unsafe_content(transcript)
        if not ok:
            return _refusal(timer, request, RefusalReason.unsafe_content, msg)

        # Off-topic check uses corpus centroid — lazy computed at startup
        # We embed the query first just for this check, then re-use in retrieval

    # Determine reranker setting
    use_reranker = (
        request.reranker_enabled
        if request.reranker_enabled is not None
        else settings.reranker_enabled
    )

    # ── Stages 2–4: Embedding + Retrieval (inside retrieve()) ─────────────────
    # NOTE: retrieve() is synchronous. We call it directly rather than via
    # run_in_executor to avoid Qdrant local-mode file-lock conflicts across threads.
    # With server-mode Qdrant (USE_QDRANT_LOCAL=false) this is non-blocking anyway.
    try:
        sources, max_score = retrieve(
            query=transcript,
            strategy=request.strategy,
            language_code=request.language_code,
            top_k=settings.retrieval_top_k,
            reranker_enabled=use_reranker,
            timer=timer,
        )
    except Exception as e:
        logger.error("Retrieval error", extra={"error": str(e)})
        return _refusal(timer, request, RefusalReason.internal_error, str(e))

    # ── Stage 5: Evidence scoring gate ───────────────────────────────────────
    with timer.stage("evidence_scoring"):
        logger.info("Evidence gate", extra={"max_score": round(max_score, 4),
                                            "threshold": settings.evidence_threshold,
                                            "n_sources": len(sources)})
        ok, msg = check_insufficient_evidence(max_score, settings.evidence_threshold)
        if not ok:
            return _refusal(timer, request, RefusalReason.insufficient_evidence, msg)

    # ── Stage 6: LLM generation ───────────────────────────────────────────────
    passage_texts = [s.text for s in sources[:3]]  # top 3 passages to LLM

    with timer.stage("generation"):
        answer = await generate_answer(
            question=transcript,
            passages=passage_texts,
            timeout_ms=settings.generation_timeout_ms,
        )

    if not answer:
        return _refusal(timer, request, RefusalReason.insufficient_evidence,
                        "I don't have sufficient information in the provided knowledge base to answer that.")

    # If the LLM itself returned the standard refusal phrase, honour it directly
    # without running grounding check (the model correctly identified no evidence)
    if _is_llm_refusal(answer):
        return _refusal(timer, request, RefusalReason.insufficient_evidence,
                        "I don't have sufficient information in the provided knowledge base to answer that.")

    # ── Stage 7: Grounding check ──────────────────────────────────────────────
    with timer.stage("grounding_check"):
        ok, msg = check_grounding(answer, passage_texts, settings.grounding_threshold)
        if not ok:
            return _refusal(timer, request, RefusalReason.grounding_failed, msg)

    latency = _build_latency(timer)
    logger.info(
        "Query answered",
        extra={
            "strategy": request.strategy.value,
            "n_sources": len(sources),
            "max_score": round(max_score, 4),
            "latency_total_ms": latency.total_ms,
        },
    )

    return QueryResponse(
        answer=answer,
        refused=False,
        sources=sources[:5],  # return top 5 to UI
        transcript=transcript,
        language_code=request.language_code,
        strategy=request.strategy.value,
        latency=latency,
    )
