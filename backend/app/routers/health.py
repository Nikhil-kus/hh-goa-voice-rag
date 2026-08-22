from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.vector_store import all_collection_info
from app.services.embedder import get_model
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    try:
        get_model()
        embed_status = "ok"
    except Exception as e:
        embed_status = f"error: {e}"

    try:
        collections = all_collection_info()
        qdrant_status = "ok"
    except Exception as e:
        qdrant_status = f"error: {e}"
        collections = {}

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        embedding_model=settings.embedding_model,
        collections=collections,
    )
