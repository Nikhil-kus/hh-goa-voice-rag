"""
Latency benchmark — measures P50/P70/P100 per pipeline stage.

Uses text queries directly (no STT) to isolate RAG pipeline latency.
STT latency is measured separately (network-dependent, not local compute).

Run from backend/:
    python scripts/benchmark_latency.py

Output:
    benchmark_latency_{timestamp}.json   (machine-readable)
    benchmark_latency_{timestamp}.txt    (human-readable)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

from app.config import settings
from app.models.schemas import ChunkingStrategy
from app.services.guardrails import (
    check_grounding,
    check_insufficient_evidence,
    check_unsafe_content,
)
from app.services.retriever import retrieve
from app.utils.logger import configure_root_logger, get_logger
from app.utils.timing import StageTimer

configure_root_logger("WARNING")  # suppress info noise during benchmark
logger = get_logger("benchmark_latency")


def load_queries(eval_path: Path, n: int = 100) -> List[Dict[str, Any]]:
    """Load benchmark queries from eval_queries.jsonl."""
    if not eval_path.exists():
        raise FileNotFoundError(
            f"{eval_path} not found. Run scripts/prepare_eval.py first."
        )
    queries = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
            if len(queries) >= n:
                break
    return queries


def percentile(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    return float(np.percentile(data, p))


def run_benchmark(
    queries: List[Dict[str, Any]],
    strategy: ChunkingStrategy,
    n_queries: int,
) -> Dict[str, Any]:
    """Run latency benchmark for one strategy."""
    print(f"\nBenchmarking strategy: {strategy.value} ({n_queries} queries)")

    stage_data: Dict[str, List[float]] = {
        "query_normalization": [],
        "embedding": [],
        "vector_retrieval": [],
        "reranking": [],
        "evidence_scoring": [],
        "total_rag": [],  # normalization+embed+retrieval (no LLM)
    }

    errors = 0
    for i, q in enumerate(queries[:n_queries]):
        timer = StageTimer()
        query_text = q.get("query_text", "")
        lang_code = q.get("language", None)

        try:
            sources, max_score = retrieve(
                query=query_text,
                strategy=strategy,
                language_code=f"{lang_code}-IN" if lang_code else None,
                top_k=settings.retrieval_top_k,
                reranker_enabled=False,
                timer=timer,
            )
            ms = timer.all_ms()

            stage_data["query_normalization"].append(ms.get("query_normalization", 0))
            stage_data["embedding"].append(ms.get("embedding", 0))
            stage_data["vector_retrieval"].append(ms.get("vector_retrieval", 0))
            stage_data["reranking"].append(ms.get("reranking", 0))

            # Evidence scoring (lightweight — just measure the call)
            t0 = time.perf_counter()
            check_insufficient_evidence(max_score, settings.evidence_threshold)
            es_ms = (time.perf_counter() - t0) * 1000
            stage_data["evidence_scoring"].append(round(es_ms, 2))

            rag_total = (
                ms.get("query_normalization", 0)
                + ms.get("embedding", 0)
                + ms.get("vector_retrieval", 0)
                + ms.get("reranking", 0)
                + es_ms
            )
            stage_data["total_rag"].append(round(rag_total, 2))

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_queries}] last rag={rag_total:.1f}ms", flush=True)

        except Exception as e:
            errors += 1
            logger.warning("Query failed", extra={"i": i, "error": str(e)})

    result: Dict[str, Any] = {
        "strategy": strategy.value,
        "n_queries": n_queries,
        "n_errors": errors,
        "stages": {},
    }

    for stage, values in stage_data.items():
        if values:
            result["stages"][stage] = {
                "p50": round(percentile(values, 50), 2),
                "p70": round(percentile(values, 70), 2),
                "p100": round(percentile(values, 100), 2),
                "mean": round(float(np.mean(values)), 2),
                "n": len(values),
            }

    return result


def main() -> None:
    eval_path = Path("eval_queries.jsonl")
    n_queries = 100
    strategies = [s for s in ChunkingStrategy]

    print(f"Loading {n_queries} queries from {eval_path}...")
    try:
        queries = load_queries(eval_path, n=n_queries)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Loaded {len(queries)} queries")

    all_results = []
    for strategy in strategies:
        result = run_benchmark(queries, strategy, min(n_queries, len(queries)))
        all_results.append(result)

    # Save outputs
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    json_path = Path(f"benchmark_latency_{ts}.json")
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Human-readable
    txt_path = Path(f"benchmark_latency_{ts}.txt")
    lines = [
        "=" * 70,
        f"LATENCY BENCHMARK  {ts}  n={n_queries} queries",
        "=" * 70,
    ]
    for r in all_results:
        lines.append(f"\nStrategy: {r['strategy']}  errors={r['n_errors']}")
        lines.append(f"  {'Stage':<22} {'P50':>8} {'P70':>8} {'P100':>8} {'Mean':>8}")
        lines.append(f"  {'-'*54}")
        for stage, s in r["stages"].items():
            lines.append(
                f"  {stage:<22} {s['p50']:>7.1f}ms {s['p70']:>7.1f}ms "
                f"{s['p100']:>7.1f}ms {s['mean']:>7.1f}ms"
            )

    txt_content = "\n".join(lines)
    txt_path.write_text(txt_content, encoding="utf-8")
    print("\n" + txt_content)
    print(f"\nResults: {json_path}  {txt_path}")


if __name__ == "__main__":
    main()
