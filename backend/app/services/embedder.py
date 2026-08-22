"""
Multilingual embedding service.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 384-dim, ~470MB, ~5-15ms single-query inference on CPU
  - Loaded once at startup and reused for all requests
  - Supports all 5 target languages (hi, bn, ta, te, kn)
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.utils.logger import get_logger
from app.utils.timing import StageTimer

logger = get_logger(__name__)

# Module-level singleton — loaded once on first use
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model", extra={"model": settings.embedding_model})
        t0 = time.perf_counter()
        _model = SentenceTransformer(settings.embedding_model)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Embedding model loaded", extra={"elapsed_ms": round(elapsed, 1)})
    return _model


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string.
    Returns a normalized float32 numpy array of shape (384,).
    Normalizing ensures cosine similarity == dot product in Qdrant.
    """
    model = get_model()
    t0 = time.perf_counter()
    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug("Query embedded", extra={"elapsed_ms": round(elapsed_ms, 2), "dims": len(vector)})
    return vector.astype(np.float32)


def embed_batch(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts in batches.
    Used by the ingestion script — not the hot path.
    Returns shape (N, 384).
    """
    model = get_model()
    t0 = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Batch embedded",
        extra={"n": len(texts), "elapsed_ms": round(elapsed_ms, 1)},
    )
    return vectors.astype(np.float32)
