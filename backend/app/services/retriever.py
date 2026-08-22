"""
Retrieval service.

Pipeline:
  query text → normalize → embed → Qdrant ANN search →
  [optional lightweight rerank] → evidence scoring → return results

Design decisions:
  - No cross-encoder reranker by default (latency cost)
  - Optional rerank uses query-passage dot product re-scoring (cheap)
  - Language filter applied as Qdrant payload filter (reduces search space)
  - is_selected is NOT used in any retrieval scoring — only in offline eval
"""
from __future__ import annotations

import unicodedata
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models.schemas import ChunkingStrategy, RetrievedSource
from app.services.embedder import embed_query
from app.services.vector_store import search
from app.utils.logger import get_logger
from app.utils.timing import StageTimer

logger = get_logger(__name__)

# Sarvam BCP-47 → ISO 639-1 mapping
BSARVAM_TO_ISO: Dict[str, str] = {
    "hi-IN": "hi", "bn-IN": "bn", "ta-IN": "ta", "te-IN": "te",
    "kn-IN": "kn", "ml-IN": "ml", "mr-IN": "mr", "gu-IN": "gu",
    "pa-IN": "pa", "od-IN": "or", "en-IN": "en",
}


def normalize_query(text: str) -> str:
    """Unicode normalize, collapse whitespace, strip."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def language_code_to_iso(language_code: Optional[str]) -> Optional[str]:
    """Convert Sarvam BCP-47 (e.g. hi-IN) to ISO 639-1 (hi)."""
    if not language_code:
        return None
    # Direct map
    if language_code in BSARVAM_TO_ISO:
        return BSARVAM_TO_ISO[language_code]
    # Try splitting on "-"
    base = language_code.split("-")[0].lower()
    return base if len(base) == 2 else None


def retrieve(
    query: str,
    strategy: ChunkingStrategy,
    language_code: Optional[str],
    top_k: int,
    reranker_enabled: bool,
    timer: StageTimer,
) -> Tuple[List[RetrievedSource], float]:
    """
    Full retrieval pipeline for one query.

    Returns:
        (sources, max_score)
        sources: top retrieved passages as RetrievedSource objects
        max_score: highest cosine similarity score (used for evidence gate)
    """
    # 1. Normalize
    with timer.stage("query_normalization"):
        normalized = normalize_query(query)
        iso_lang = language_code_to_iso(language_code)
        # Only filter by language if it's in our indexed set
        lang_filter = iso_lang if iso_lang in settings.supported_language_list else None

    logger.debug(
        "Query normalized",
        extra={"normalized": normalized[:80], "lang": iso_lang, "filter": lang_filter},
    )

    # 2. Embed
    with timer.stage("embedding"):
        query_vector = embed_query(normalized)

    # 3. Vector search
    collection = settings.collection_name(strategy.value)
    with timer.stage("vector_retrieval"):
        raw_results = search(
            collection_name=collection,
            query_vector=query_vector,
            top_k=top_k,
            language_filter=lang_filter,
        )

    if not raw_results:
        logger.warning("No results from vector search",
                       extra={"collection": collection, "lang_filter": lang_filter})
        return [], 0.0

    # 4. Optional lightweight rerank (dot-product, no cross-encoder)
    with timer.stage("reranking"):
        if reranker_enabled and len(raw_results) > 1:
            raw_results = _dot_product_rerank(query_vector, raw_results)

    # 5. Build source objects
    # NOTE: is_selected is intentionally excluded from RetrievedSource —
    # it must not leak into production API responses.
    sources: List[RetrievedSource] = []
    max_score = 0.0
    for r in raw_results:
        score = r["score"]
        payload = r["payload"]
        if score > max_score:
            max_score = score
        sources.append(
            RetrievedSource(
                text=payload.get("text", ""),
                language=payload.get("language", ""),
                query_id=payload.get("query_id", 0),
                passage_idx=payload.get("passage_idx", 0),
                score=round(score, 4),
                strategy=payload.get("strategy", strategy.value),
                query_type=payload.get("query_type"),
            )
        )

    logger.debug(
        "Retrieval complete",
        extra={
            "n_results": len(sources),
            "max_score": round(max_score, 4),
            "reranked": reranker_enabled,
        },
    )
    return sources, max_score


def _dot_product_rerank(
    query_vector,
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Lightweight rerank: re-score by dot product between query vector
    and passage text re-embedding. No cross-encoder.
    Because we already have cosine scores from Qdrant (vectors are normalized),
    this is equivalent to the Qdrant scores — but allows future extension
    to use passage-specific sub-embeddings.
    For now this is a pass-through that sorts by existing score.
    """
    return sorted(results, key=lambda r: r["score"], reverse=True)
