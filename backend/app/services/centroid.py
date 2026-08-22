"""
Corpus centroid for off-topic query detection.

Computes the mean of a sample of passage embeddings from Qdrant
(or a small fixed set of seed sentences) to represent the "on-topic" center.

The centroid is stored as a normalized L2 vector.
At query time, cosine_similarity(query_embedding, centroid) < threshold
indicates the query is likely off-topic.

Computed lazily on first use and cached in memory.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

_centroid: Optional[np.ndarray] = None
_lock = threading.Lock()

# Seed sentences representative of MS MARCO / general knowledge Q&A
# Used as fallback if Qdrant is unavailable or empty
_SEED_SENTENCES = [
    "what is the capital of India",
    "how does photosynthesis work",
    "who was the first president of the united states",
    "what causes earthquakes",
    "explain the water cycle",
    "what is the manhattan project",
    "describe the french revolution",
    "how do vaccines work",
    "what is machine learning",
    "explain supply and demand",
    "who wrote hamlet",
    "what is the speed of light",
    "how is steel made",
    "what are the causes of world war 2",
    "describe the human immune system",
    "what is quantum mechanics",
    "how does the internet work",
    "what is the gross domestic product",
    "explain natural selection",
    "what is the significance of the magna carta",
]


def get_centroid() -> Optional[np.ndarray]:
    """Return cached centroid, computing it lazily on first call."""
    global _centroid
    if _centroid is not None:
        return _centroid
    with _lock:
        if _centroid is not None:
            return _centroid
        _centroid = _compute_centroid()
    return _centroid


def _compute_centroid() -> Optional[np.ndarray]:
    """
    Compute centroid from seed sentences.
    Using seed sentences is simpler than sampling from Qdrant and avoids
    a Qdrant dependency at startup; it correctly represents QA-style queries.
    """
    try:
        from app.services.embedder import embed_batch
        logger.info("Computing corpus centroid from seed sentences...")
        vecs = embed_batch(_SEED_SENTENCES, batch_size=len(_SEED_SENTENCES))
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        logger.info("Corpus centroid computed", extra={"dim": centroid.shape[0]})
        return centroid.astype(np.float32)
    except Exception as e:
        logger.warning("Could not compute centroid, off-topic check disabled",
                       extra={"error": str(e)})
        return None


def reset_centroid() -> None:
    """Force recomputation on next access (useful for testing)."""
    global _centroid
    with _lock:
        _centroid = None
