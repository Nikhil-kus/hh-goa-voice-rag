"""
Qdrant abstraction layer.

Supports two modes:
  - Local (embedded): qdrant-client stores index on disk at qdrant_local_path
    Used for development and when Docker is not available.
  - Server: connects to a running Qdrant instance (Docker / cloud)

Set USE_QDRANT_LOCAL=true in .env for local mode.
Set USE_QDRANT_LOCAL=false and provide QDRANT_HOST/PORT for server mode.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if settings.use_qdrant_local:
            logger.info(
                "Using Qdrant local (embedded) mode",
                extra={"path": settings.qdrant_local_path},
            )
            _client = QdrantClient(path=settings.qdrant_local_path)
        else:
            logger.info(
                "Connecting to Qdrant server",
                extra={"host": settings.qdrant_host, "port": settings.qdrant_port},
            )
            kwargs: Dict[str, Any] = {
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
            }
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            _client = QdrantClient(**kwargs)
    return _client


def ensure_collection(collection_name: str, recreate: bool = False) -> None:
    """
    Create the collection if it does not exist.
    Pass recreate=True during ingestion to start fresh.
    """
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        if recreate:
            logger.info("Recreating collection", extra={"collection": collection_name})
            client.delete_collection(collection_name)
        else:
            logger.info("Collection exists", extra={"collection": collection_name})
            return

    logger.info("Creating collection", extra={"collection": collection_name})
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=settings.embedding_dim,
            distance=Distance.COSINE,
            on_disk=False,          # keep vectors in RAM for low latency
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
            full_scan_threshold=10_000,
        ),
    )

    # Index payload fields used for filtering
    for field_name, field_schema in [
        ("language",    qmodels.PayloadSchemaType.KEYWORD),
        ("query_type",  qmodels.PayloadSchemaType.KEYWORD),
        ("strategy",    qmodels.PayloadSchemaType.KEYWORD),
        ("query_id",    qmodels.PayloadSchemaType.INTEGER),
        ("is_selected", qmodels.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
    logger.info("Collection ready", extra={"collection": collection_name})


def upsert_points(
    collection_name: str,
    vectors: np.ndarray,
    payloads: List[Dict[str, Any]],
    batch_size: int = 256,
) -> int:
    """
    Upsert vectors + payloads to Qdrant.
    Returns total points upserted.
    """
    client = get_client()
    total = 0
    for start in range(0, len(vectors), batch_size):
        batch_vecs = vectors[start : start + batch_size]
        batch_pays = payloads[start : start + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload=pay,
            )
            for vec, pay in zip(batch_vecs, batch_pays)
        ]
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
    return total


def search(
    collection_name: str,
    query_vector: np.ndarray,
    top_k: int = 10,
    language_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    ANN search in Qdrant.

    Args:
        collection_name: which strategy collection to search
        query_vector:    normalized float32 (384,)
        top_k:           number of results to return
        language_filter: ISO 639-1 code (e.g. "hi") — applied as Qdrant filter
                         to reduce search space. None = search all languages.

    Returns:
        list of dicts with keys: score, payload
    """
    client = get_client()

    query_filter: Optional[Filter] = None
    if language_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="language",
                    match=MatchValue(value=language_filter),
                )
            ]
        )

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {"score": float(hit.score), "payload": hit.payload or {}}
        for hit in results
    ]


def collection_info(collection_name: str) -> Dict[str, Any]:
    client = get_client()
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }
    except Exception as e:
        return {"name": collection_name, "error": str(e)}


def all_collection_info() -> Dict[str, Dict[str, Any]]:
    return {
        settings.collection_name(s): collection_info(settings.collection_name(s))
        for s in settings.strategy_list
    }
