"""
Offline ingestion script.

Loads MSMARCO-XI from locally downloaded Parquet files (run download_data.py first),
applies all 3 chunking strategies, embeds with multilingual-MiniLM, upserts to Qdrant.

Dataset layout (train parquets per language):
  data/msmarco_xi/train/{prefix}train.parquet
  data/msmarco_xi/validation/{prefix}val.parquet
  prefixes: hin, ben, tam, kan, mal, mar, guj, pan, ori, asm, nep, urd, san
  NOTE: Telugu (tel) has validation only — no train parquet.

Run from backend/:
    python scripts/ingest.py

Env vars:
    LANGUAGES_TO_INDEX=hi,bn,ta,te,kn
    RECORDS_PER_LANGUAGE=1000
    CHUNKING_STRATEGIES=fixed_size_overlap,sentence_aware,passage_structure_aware

IMPORTANT invariants (enforced):
  - Only passage text embedded (never answer, never query)
  - is_selected stored as payload metadata only, never used for ranking
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from app.config import settings
from app.services.chunker import chunk_record
from app.services.embedder import embed_batch
from app.services.vector_store import ensure_collection, upsert_points
from app.utils.logger import configure_root_logger, get_logger

configure_root_logger(settings.log_level)
logger = get_logger("ingest")

DATA_DIR = Path("data/msmarco_xi")

# We index from validation parquets (~440MB each, practical to download).
# Train parquets are ~3.5GB each — too large for this environment.
# Documented limitation: index covers val split; architecture scales to train when compute allows.
LANG_TO_VAL_PREFIX: Dict[str, str] = {
    "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
    "kn": "kan", "ml": "mal", "mr": "mar", "gu": "guj",
    "pa": "pan", "or": "ori", "as": "asm", "ne": "nep",
    "ur": "urd", "sa": "san",
}

UPSERT_BATCH = 64   # smaller batches for Qdrant Cloud free tier
EMBED_BATCH  = 32


def iter_parquet(path: Path, max_records: int):
    """
    Yield records from a Parquet file up to max_records.
    Uses PyArrow batch reading to avoid loading full file into RAM.
    """
    pf = pq.ParquetFile(path)
    yielded = 0
    for batch in pf.iter_batches(batch_size=512):
        table = batch.to_pydict()
        n = len(table["query_id"])
        for i in range(n):
            if yielded >= max_records:
                return
            record = {k: table[k][i] for k in table}
            # passages is stored as a struct — convert to dict
            passages_raw = record.get("passages", {})
            if hasattr(passages_raw, "as_py"):
                passages_raw = passages_raw.as_py()
            record["passages"] = passages_raw
            yield record
            yielded += 1


def process_language_strategy(
    lang: str,
    strategy: str,
    records_per_language: int,
    recreate: bool = False,
) -> Dict[str, Any]:
    prefix = LANG_TO_VAL_PREFIX.get(lang)
    if not prefix:
        msg = f"Unknown language: {lang}"
        logger.warning(msg)
        return {"lang": lang, "strategy": strategy, "skipped": True, "reason": msg}

    # Use validation parquet for indexing (train files are too large to download)
    parquet_path = DATA_DIR / "validation" / f"{prefix}val.parquet"
    if not parquet_path.exists():
        msg = f"File not found: {parquet_path}. Run scripts/download_data.py first."
        logger.error(msg)
        return {"lang": lang, "strategy": strategy, "error": msg}

    collection = settings.collection_name(strategy)
    logger.info(
        "Processing",
        extra={"lang": lang, "strategy": strategy, "collection": collection,
               "file": str(parquet_path)},
    )

    ensure_collection(collection, recreate=recreate)

    records_used = 0
    chunks_created = 0
    vectors_upserted = 0
    t_start = time.perf_counter()

    pending_texts: List[str] = []
    pending_payloads: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal vectors_upserted
        if not pending_texts:
            return
        vecs = embed_batch(pending_texts, batch_size=EMBED_BATCH)
        n = upsert_points(collection, vecs, pending_payloads, batch_size=UPSERT_BATCH)
        vectors_upserted += n
        pending_texts.clear()
        pending_payloads.clear()

    FLUSH_EVERY = 500

    for record in iter_parquet(parquet_path, max_records=records_per_language):
        passages = record.get("passages") or {}
        if not passages or not passages.get("English_passages"):
            continue

        chunks = chunk_record(record, strategy=strategy, source="english")
        if not chunks:
            continue

        records_used += 1
        for chunk in chunks:
            pending_texts.append(chunk.text)
            pending_payloads.append(chunk.to_payload())
            chunks_created += 1

        if len(pending_texts) >= FLUSH_EVERY:
            flush()
            logger.info(
                "Progress",
                extra={"lang": lang, "strategy": strategy,
                       "records": records_used, "chunks": chunks_created,
                       "upserted": vectors_upserted},
            )

    flush()

    elapsed = time.perf_counter() - t_start
    summary = {
        "lang": lang,
        "strategy": strategy,
        "collection": collection,
        "records_used": records_used,
        "chunks_created": chunks_created,
        "vectors_upserted": vectors_upserted,
        "elapsed_s": round(elapsed, 1),
    }
    logger.info("Done", extra=summary)
    return summary


def main() -> None:
    languages = settings.language_list
    strategies = settings.strategy_list
    records_per_lang = settings.records_per_language

    supported = [l for l in languages if l in LANG_TO_VAL_PREFIX]
    skipped   = [l for l in languages if l not in LANG_TO_VAL_PREFIX]
    if skipped:
        logger.warning("No train parquet for", extra={"langs": skipped})

    logger.info(
        "Starting ingestion",
        extra={"languages": supported, "strategies": strategies,
               "records_per_language": records_per_lang},
    )

    all_summaries: List[Dict[str, Any]] = []

    for strategy in strategies:
        for lang_idx, lang in enumerate(supported):
            recreate = (lang_idx == 0)
            try:
                s = process_language_strategy(lang, strategy, records_per_lang, recreate)
                all_summaries.append(s)
            except Exception as e:
                logger.error("Failed", extra={"lang": lang, "strategy": strategy, "error": str(e)})
                all_summaries.append({"lang": lang, "strategy": strategy, "error": str(e)})

    report = Path("ingestion_report.json")
    report.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written", extra={"path": str(report)})

    print("\n" + "=" * 72)
    print("INGESTION SUMMARY")
    print("=" * 72)
    total_chunks = total_vecs = 0
    for s in all_summaries:
        if "error" in s:
            print(f"  ERROR  lang={s.get('lang')}  strategy={s.get('strategy')}: {s['error']}")
        elif s.get("skipped"):
            print(f"  SKIP   lang={s.get('lang')}  {s.get('reason')}")
        else:
            print(
                f"  OK  {s['lang']:4s}  {s['strategy']:28s}  "
                f"records={s['records_used']:5d}  chunks={s['chunks_created']:7d}  "
                f"vecs={s['vectors_upserted']:7d}  {s['elapsed_s']}s"
            )
            total_chunks += s.get("chunks_created", 0)
            total_vecs   += s.get("vectors_upserted", 0)
    print("-" * 72)
    print(f"  TOTAL  chunks={total_chunks}  vectors={total_vecs}")
    print("=" * 72)


if __name__ == "__main__":
    main()
