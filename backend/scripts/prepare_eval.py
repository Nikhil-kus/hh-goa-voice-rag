"""
Build evaluation query set from MSMARCO-XI validation splits.

Uses direct Parquet file URLs (dataset has no per-language configs).
Validation splits available: hin, ben, tam, kan, mal, mar, guj, pan,
                              ori, asm, nep, urd, san, tel

Saves eval_queries.jsonl — one JSON line per query with:
  query_id, query_text (English), language, query_type,
  and the passage list with is_selected ground truth.

is_selected kept here for offline Recall@K evaluation ONLY.
Never used in live retrieval.

Run from backend/:
    python scripts/prepare_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from app.config import settings
from app.utils.logger import configure_root_logger, get_logger

configure_root_logger(settings.log_level)
logger = get_logger("prepare_eval")

LANG_TO_VAL_PREFIX: Dict[str, str] = {
    "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
    "kn": "kan", "ml": "mal", "mr": "mar", "gu": "guj",
    "pa": "pan", "or": "ori", "as": "asm", "ne": "nep",
    "ur": "urd", "sa": "san",
}

DATA_DIR = Path("data/msmarco_xi")
MAX_EVAL_PER_LANG = 200


def iter_val_parquet(path: Path, max_records: int):
    pf = pq.ParquetFile(path)
    yielded = 0
    for batch in pf.iter_batches(batch_size=256):
        table = batch.to_pydict()
        n = len(table["query_id"])
        for i in range(n):
            if yielded >= max_records:
                return
            record = {k: table[k][i] for k in table}
            passages_raw = record.get("passages", {})
            if hasattr(passages_raw, "as_py"):
                passages_raw = passages_raw.as_py()
            record["passages"] = passages_raw
            yield record
            yielded += 1


def process_language(lang: str) -> List[Dict[str, Any]]:
    prefix = LANG_TO_VAL_PREFIX.get(lang)
    if not prefix:
        logger.warning("No val parquet for language", extra={"lang": lang})
        return []

    val_path = DATA_DIR / "validation" / f"{prefix}val.parquet"
    if not val_path.exists():
        logger.error("Val file not found. Run download_data.py first.",
                     extra={"path": str(val_path)})
        return []

    logger.info("Reading val split", extra={"lang": lang, "path": str(val_path)})

    queries: List[Dict[str, Any]] = []
    for record in iter_val_parquet(val_path, max_records=MAX_EVAL_PER_LANG * 5):
        if len(queries) >= MAX_EVAL_PER_LANG:
            break

        passages = record.get("passages", {})
        eng_passages: List[str] = passages.get("English_passages", []) or []
        is_selected: List[int] = passages.get("is_selected", []) or []

        if not eng_passages:
            continue
        if not any(s == 1 for s in is_selected):
            continue

        query_id  = int(record.get("query_id", 0))
        query_text = str(record.get("Eng_Query") or record.get("query") or "")
        query_type = str(record.get("query_type") or "UNKNOWN")

        passage_list = []
        for idx, (text, sel) in enumerate(zip(eng_passages, is_selected)):
            if text and text.strip():
                passage_list.append({
                    "text": text.strip(),
                    "is_selected": int(sel),
                    "passage_idx": idx,
                })

        if not passage_list:
            continue

        queries.append({
            "query_id":   query_id,
            "query_text": query_text,
            "language":   lang,
            "query_type": query_type,
            "passages":   passage_list,
        })

    logger.info("Val queries collected", extra={"lang": lang, "count": len(queries)})
    return queries


def main() -> None:
    languages = settings.language_list
    all_queries: List[Dict[str, Any]] = []

    for lang in languages:
        queries = process_language(lang)
        all_queries.extend(queries)

    out_path = Path("eval_queries.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info("Eval set written", extra={"path": str(out_path), "total": len(all_queries)})
    print(f"\nEval set: {len(all_queries)} queries → {out_path}")
    lang_counts: Dict[str, int] = {}
    for q in all_queries:
        lang_counts[q["language"]] = lang_counts.get(q["language"], 0) + 1
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count} queries")


if __name__ == "__main__":
    main()
