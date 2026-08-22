"""
Guardrail checks — run before and after the RAG pipeline.

Pre-query checks (block before any embedding/retrieval):
  1. empty_transcript     — no text at all
  2. unsafe_content       — regex blocklist for harmful content
  3. off_topic            — embedding similarity vs corpus centroid

Evidence gate (after retrieval, before generation):
  4. insufficient_evidence — max retrieval score below threshold

Post-generation check:
  5. grounding_check       — ROUGE-1 recall between answer and retrieved passages

All checks return (passed: bool, reason: Optional[str], message: Optional[str]).
Caller decides what to do on failure.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Unsafe content patterns ───────────────────────────────────────────────────
_UNSAFE_PATTERNS = [
    # Violence / self-harm
    r"\b(how to (make|build|create) (a |an )?(bomb|explosive|weapon|poison|drug))\b",
    r"\b(suicide|self.?harm|kill (myself|yourself|himself|herself))\b",
    r"\b(child (porn|sex|abuse|exploit))\b",
    r"\b(terrorist|terrorism|jihad|radicali[sz])\b",
    # Hate speech
    r"\b(n[i1]gg[ae]r|fagg[oa]t|ch[i1]nk|sp[i1]c)\b",
]
_UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]

# ── ROUGE-1 helper ────────────────────────────────────────────────────────────

def _rouge1_recall(hypothesis: str, reference: str) -> float:
    """Compute unigram recall: |hyp_tokens ∩ ref_tokens| / |ref_tokens|."""
    hyp_tokens = set(hypothesis.lower().split())
    ref_tokens  = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = hyp_tokens & ref_tokens
    return len(overlap) / len(ref_tokens)


# ── Check functions ───────────────────────────────────────────────────────────

def check_empty_transcript(transcript: str) -> Tuple[bool, Optional[str]]:
    """Return (ok, message). ok=False means fail."""
    if not transcript or not transcript.strip():
        return False, "No speech detected. Please try speaking again."
    if len(transcript.strip()) < 3:
        return False, "Transcript too short. Please speak more clearly."
    return True, None


def check_unsafe_content(text: str) -> Tuple[bool, Optional[str]]:
    """Return (ok, message). ok=False means unsafe content detected."""
    for pattern in _UNSAFE_RE:
        if pattern.search(text):
            logger.warning("Unsafe content detected", extra={"text_prefix": text[:60]})
            return False, "I'm not able to help with that request."
    return True, None


def check_off_topic(
    query_embedding,
    corpus_centroid,
    threshold: float,
) -> Tuple[bool, Optional[str]]:
    """
    Compare query embedding to pre-computed corpus centroid.
    If cosine similarity < threshold → off-topic.

    Both vectors must be L2-normalized (as produced by embed_query).
    Cosine similarity == dot product for normalized vectors.
    """
    if corpus_centroid is None:
        # No centroid computed yet — skip check
        return True, None

    import numpy as np
    similarity = float(np.dot(query_embedding, corpus_centroid))
    logger.debug("Off-topic check", extra={"similarity": round(similarity, 4), "threshold": threshold})
    if similarity < threshold:
        return False, (
            "Your query appears to be outside the scope of this knowledge base. "
            "Please ask questions related to the available topics."
        )
    return True, None


def check_insufficient_evidence(
    max_score: float,
    threshold: float,
) -> Tuple[bool, Optional[str]]:
    """Return (ok, message). ok=False means not enough evidence."""
    if max_score < threshold:
        logger.info(
            "Insufficient evidence",
            extra={"max_score": round(max_score, 4), "threshold": threshold},
        )
        return False, (
            "I don't have sufficient information in the provided knowledge base "
            "to answer that."
        )
    return True, None


def check_grounding(
    answer: str,
    sources: List[str],
    threshold: float,
) -> Tuple[bool, Optional[str]]:
    """
    Verify the generated answer is grounded in retrieved passages.
    Uses ROUGE-1 recall between answer and concatenated source texts.
    ok=False means answer is not sufficiently supported by evidence.
    """
    if not answer or not sources:
        return False, "I don't have sufficient information in the provided knowledge base to answer that."

    reference = " ".join(sources)
    recall = _rouge1_recall(answer, reference)

    logger.debug(
        "Grounding check",
        extra={"rouge1_recall": round(recall, 4), "threshold": threshold},
    )

    if recall < threshold:
        logger.warning(
            "Answer not sufficiently grounded",
            extra={"recall": round(recall, 4), "threshold": threshold},
        )
        return False, (
            "I don't have sufficient information in the provided knowledge base "
            "to answer that."
        )
    return True, None


def check_max_length(transcript: str, max_chars: int) -> str:
    """Truncate transcript if it exceeds max length. Returns (possibly truncated) text."""
    if len(transcript) > max_chars:
        logger.warning(
            "Transcript truncated",
            extra={"original_len": len(transcript), "max_chars": max_chars},
        )
        return transcript[:max_chars]
    return transcript
