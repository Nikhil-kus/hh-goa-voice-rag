"""
FastAPI application entry point.
Loads embedding model, Qdrant client, and corpus centroid at startup.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import benchmark, health, query, transcribe
from app.services.centroid import get_centroid
from app.services.embedder import get_model
from app.services.vector_store import get_client
from app.utils.logger import configure_root_logger, get_logger

configure_root_logger(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup — loading embedding model...")
    get_model()
    logger.info("Startup — connecting to Qdrant...")
    try:
        get_client()
        logger.info("Qdrant connected.")
    except Exception as e:
        logger.warning("Qdrant unavailable at startup", extra={"error": str(e)})
    logger.info("Startup — computing corpus centroid...")
    get_centroid()
    logger.info("Ready.")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Voice RAG System",
    description="HH Goa 2026 — Voice-Enabled RAG over MSMARCO-XI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(transcribe.router)
app.include_router(query.router)
app.include_router(benchmark.router)
