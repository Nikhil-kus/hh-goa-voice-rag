"""
Benchmark router — exposes latency and retrieval quality benchmarks via API.
Results are saved to disk and returned as structured JSON.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.models.schemas import BenchmarkRequest, BenchmarkResult, StagePct
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Directory where benchmark scripts write results
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class BenchmarkStatus(BaseModel):
    status: str                          # "running" | "complete" | "error"
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


def _load_latest_result(prefix: str) -> Optional[Dict[str, Any]]:
    """Load the most recent benchmark JSON output file."""
    files = sorted(BACKEND_DIR.glob(f"{prefix}_*.json"), reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


@router.post("/api/benchmark/latency", response_model=BenchmarkStatus)
async def run_latency_benchmark(background_tasks: BackgroundTasks):
    """
    Trigger latency benchmark in background.
    Returns immediately; poll GET /api/benchmark/latency/results for output.
    """
    def _run():
        try:
            result = subprocess.run(
                [sys.executable, "scripts/benchmark_latency.py"],
                cwd=str(BACKEND_DIR),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error("Latency benchmark failed", extra={"stderr": result.stderr[-500:]})
            else:
                logger.info("Latency benchmark complete")
        except Exception as e:
            logger.error("Latency benchmark exception", extra={"error": str(e)})

    background_tasks.add_task(_run)
    return BenchmarkStatus(status="running", message="Latency benchmark started")


@router.post("/api/benchmark/retrieval", response_model=BenchmarkStatus)
async def run_retrieval_benchmark(background_tasks: BackgroundTasks):
    """Trigger retrieval quality benchmark in background."""
    def _run():
        try:
            result = subprocess.run(
                [sys.executable, "scripts/benchmark_retrieval.py"],
                cwd=str(BACKEND_DIR),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error("Retrieval benchmark failed", extra={"stderr": result.stderr[-500:]})
            else:
                logger.info("Retrieval benchmark complete")
        except Exception as e:
            logger.error("Retrieval benchmark exception", extra={"error": str(e)})

    background_tasks.add_task(_run)
    return BenchmarkStatus(status="running", message="Retrieval benchmark started")


@router.get("/api/benchmark/latency/results")
async def get_latency_results() -> Dict[str, Any]:
    """Return the most recent latency benchmark results."""
    result = _load_latest_result("benchmark_latency")
    if result is None:
        raise HTTPException(status_code=404, detail="No latency benchmark results found. Run POST /api/benchmark/latency first.")
    return {"status": "complete", "result": result}


@router.get("/api/benchmark/retrieval/results")
async def get_retrieval_results() -> Dict[str, Any]:
    """Return the most recent retrieval benchmark results."""
    result = _load_latest_result("benchmark_retrieval")
    if result is None:
        raise HTTPException(status_code=404, detail="No retrieval benchmark results found. Run POST /api/benchmark/retrieval first.")
    return {"status": "complete", "result": result}


@router.get("/api/benchmark/all")
async def get_all_results() -> Dict[str, Any]:
    """Return latest results for both benchmarks if available."""
    return {
        "latency": _load_latest_result("benchmark_latency"),
        "retrieval": _load_latest_result("benchmark_retrieval"),
    }
