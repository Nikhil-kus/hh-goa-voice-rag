"""
Retrieval quality benchmark — computes Recall@K per strategy using is_selected.

For each query in eval_queries.jsonl:
  1. Embed the query
  2. Search the Qdrant collection for strategy
  3. Check if any result's query_id matches AND is from the right passage
     (passage_idx in the is_selected=1 set)
  4. Accumulate Recall@1/3/5/10

is_selected is used ONLY here in offline evaluation.
It is never used during live retrieval.

Run from backend/:
    python scripts/benchmark_retrieval.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np

from app.config import settings
from app.models.schemas import ChunkingStrategy
from app.services.embedder import embed_query
from app.services.retriever import normalize_query
from app.services.vector_store import search
from app.utils.logger import configure_root_logger, get_logger

configure_root_logger("WARNING")
logger = get_logger("benchmark_retrieval")

K_VALUES = [1, 3, 5, 10]


def load_eval_queries(path: Path, max_per_lang: int = 40) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run prepare_eval.py first.")
    queries = []
    lang_counts: Dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            lang = q.get("language", "?")
            if lang_counts[lang] < max_per_lang:
                queries.append(q)
                lang_counts[lang] += 1
    return queries


def get_selected_passage_indices(query: Dict[str, Any]) -> Set[int]:
    """Return the set of passage_idx values where is_selected==1."""
    return {
        p["passage_idx"]
        for p in query.get("passages", [])
        if p.get("is_selected") == 1
    }


def recall_at_k(
    results: List[Dict[str, Any]],
    query_id: int,
    selected_indices: Set[int],
    k: int,
) -> float:
    """
    Recall@K: 1.0 if any of the top-K results has matching query_id
    AND its passage_idx is in selected_indices. 0.0 otherwise.

    This is the standard MS MARCO retrieval metric formulation.
    """
    for r in results[:k]:
        payload = r.get("payload", {})
        if (payload.get("query_id") == query_id
                and payload.get("passage_idx") in selected_indices):
            return 1.0
    return 0.0


def benchmark_strategy(
    strategy: ChunkingStrategy,
    queries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    collection = settings.collection_name(strategy.value)
    print(f"\nEvaluating {strategy.value} on {len(queries)} queries...")

    recall_scores: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    errors = 0
    lang_counts: Dict[str, int] = defaultdict(int)

    for i, q in enumerate(queries):
        query_id = q["query_id"]
        query_text = q.get("query_text", "")
        lang = q.get("language", "")
        selected_indices = get_selected_passage_indices(q)

        if not selected_indices:
            continue  # skip queries with no positive passage

        try:
            normalized = normalize_query(query_text)
            vector = embed_query(normalized)

            # Search with language filter (same as production)
            lang_filter = lang if lang in settings.supported_language_list else None
            results = search(
                collection_name=collection,
                query_vector=vector,
                top_k=max(K_VALUES),
                language_filter=lang_filter,
            )

            for k in K_VALUES:
                r = recall_at_k(results, query_id, selected_indices, k)
                recall_scores[k].append(r)

            lang_counts[lang] += 1

            if (i + 1) % 20 == 0:
                r5 = np.mean(recall_scores[5]) if recall_scores[5] else 0
                print(f"  [{i+1}/{len(queries)}] running Recall@5={r5:.3f}", flush=True)

        except Exception as e:
            errors += 1
            logger.warning("Query failed", extra={"i": i, "error": str(e)})

    result: Dict[str, Any] = {
        "strategy": strategy.value,
        "n_queries": len(queries),
        "n_evaluated": sum(len(v) for v in recall_scores.values()) // len(K_VALUES),
        "n_errors": errors,
        "languages_evaluated": dict(lang_counts),
        "recall": {},
    }
    for k in K_VALUES:
        scores = recall_scores[k]
        if scores:
            result["recall"][f"recall_at_{k}"] = round(float(np.mean(scores)), 4)

    return result


def main() -> None:
    eval_path = Path("eval_queries.jsonl")
    max_per_lang = 40  # up to 40 per language = 160 total across 4 languages

    print(f"Loading eval queries from {eval_path}...")
    try:
        queries = load_eval_queries(eval_path, max_per_lang=max_per_lang)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    lang_dist = defaultdict(int)
    for q in queries:
        lang_dist[q["language"]] += 1
    print(f"Loaded {len(queries)} queries: {dict(lang_dist)}")

    strategies = list(ChunkingStrategy)
    all_results = []
    for strategy in strategies:
        try:
            result = benchmark_strategy(strategy, queries)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR for {strategy.value}: {e}")
            all_results.append({"strategy": strategy.value, "error": str(e)})

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    json_path = Path(f"benchmark_retrieval_{ts}.json")
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    txt_lines = [
        "=" * 65,
        f"RETRIEVAL QUALITY BENCHMARK  {ts}",
        "=" * 65,
        f"  {'Strategy':<30} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7}",
        f"  {'-'*55}",
    ]
    for r in all_results:
        if "error" in r:
            txt_lines.append(f"  {r['strategy']:<30} ERROR: {r['error']}")
        else:
            rec = r.get("recall", {})
            txt_lines.append(
                f"  {r['strategy']:<30} "
                f"{rec.get('recall_at_1', 0):>7.4f} "
                f"{rec.get('recall_at_3', 0):>7.4f} "
                f"{rec.get('recall_at_5', 0):>7.4f} "
                f"{rec.get('recall_at_10', 0):>7.4f}"
            )

    txt_content = "\n".join(txt_lines)
    txt_path = Path(f"benchmark_retrieval_{ts}.txt")
    txt_path.write_text(txt_content, encoding="utf-8")
    print("\n" + txt_content)
    print(f"\nResults: {json_path}  {txt_path}")


if __name__ == "__main__":
    main()
