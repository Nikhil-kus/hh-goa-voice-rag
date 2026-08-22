from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.vector_store import all_collection_info
from app.services.embedder import _model
from app.config import settings

router = APIRouter()


@router.get("/ping")
async def ping():
    """Ultra-fast liveness check — no dependencies."""
    return {"status": "ok"}


@router.get("/health", response_model=HealthResponse)
async def health():
    # Responds immediately even while model is loading on cold start
    try:
        collections = all_collection_info()
        qdrant_status = "ok"
    except Exception as e:
        qdrant_status = f"unavailable: {str(e)[:60]}"
        collections = {}

    embed_status = "loaded" if _model is not None else "loading"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        embedding_model=f"{settings.embedding_model} ({embed_status})",
        collections=collections,
    )
