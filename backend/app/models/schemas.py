"""
All Pydantic I/O models for the API.
Single source of truth for request/response shapes.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class ChunkingStrategy(str, Enum):
    fixed_size_overlap = "fixed_size_overlap"
    sentence_aware = "sentence_aware"
    passage_structure_aware = "passage_structure_aware"


class RefusalReason(str, Enum):
    insufficient_evidence = "insufficient_evidence"
    grounding_failed = "grounding_failed"
    off_topic = "off_topic"
    unsafe_content = "unsafe_content"
    empty_transcript = "empty_transcript"
    stt_failed = "stt_failed"
    unsupported_language = "unsupported_language"
    timeout = "timeout"
    internal_error = "internal_error"


# ── STT ───────────────────────────────────────────────────────────────────────

class TranscribeResponse(BaseModel):
    transcript: str
    language_code: str
    latency_ms: Dict[str, float]
    request_id: Optional[str] = None


# ── Retrieved source ──────────────────────────────────────────────────────────

class RetrievedSource(BaseModel):
    text: str
    language: str
    query_id: int
    passage_idx: int
    score: float
    strategy: str
    query_type: Optional[str] = None
    # is_selected is intentionally NOT exposed in live responses
    # It is only used in offline evaluation scripts


# ── Latency breakdown ─────────────────────────────────────────────────────────

class LatencyBreakdown(BaseModel):
    query_normalization_ms: Optional[float] = None
    guardrail_pre_ms: Optional[float] = None
    embedding_ms: Optional[float] = None
    vector_retrieval_ms: Optional[float] = None
    reranking_ms: Optional[float] = None
    evidence_scoring_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    grounding_check_ms: Optional[float] = None
    total_ms: float


# ── Query request/response ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=500)
    language_code: Optional[str] = Field(
        default=None,
        description="BCP-47 language code from Sarvam STT (e.g. hi-IN). "
                    "If None the system will attempt auto-detection.",
    )
    strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.passage_structure_aware,
        description="Which chunking strategy collection to query.",
    )
    reranker_enabled: Optional[bool] = Field(
        default=None,
        description="Override the server-side reranker_enabled setting.",
    )


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    refused: bool = False
    refusal_reason: Optional[RefusalReason] = None
    refusal_message: Optional[str] = None
    sources: List[RetrievedSource] = Field(default_factory=list)
    transcript: str
    language_code: Optional[str] = None
    strategy: str
    latency: LatencyBreakdown


# ── Benchmark ─────────────────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    n_queries: int = Field(default=100, ge=10, le=500)
    strategy: ChunkingStrategy = ChunkingStrategy.passage_structure_aware
    languages: Optional[List[str]] = None   # None = all indexed languages


class StagePct(BaseModel):
    p50: float
    p70: float
    p100: float


class BenchmarkResult(BaseModel):
    n_queries: int
    strategy: str
    stages: Dict[str, StagePct]   # key = stage name
    retrieval_quality: Optional[Dict[str, float]] = None  # Recall@K
    generated_at: str
    raw_path: Optional[str] = None   # path to JSON on disk


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embedding_model: str
    collections: Dict[str, Any]
