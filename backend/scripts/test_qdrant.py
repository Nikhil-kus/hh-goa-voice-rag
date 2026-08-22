"""Smoke test Qdrant local mode: create collection, upsert, search, verify."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from app.services.vector_store import (
    ensure_collection, upsert_points, search, collection_info, get_client
)
from app.config import settings

COLL = "smoke_test_collection"

print(f"Qdrant mode: {'local' if settings.use_qdrant_local else 'server'}")
print(f"Local path:  {settings.qdrant_local_path}")

# 1. Create collection
print("\n1. Creating collection...")
ensure_collection(COLL, recreate=True)
print("   OK")

# 2. Upsert 10 random vectors
print("2. Upserting 10 points...")
rng = np.random.default_rng(42)
vecs = rng.random((10, 384)).astype(np.float32)
# normalize
vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
payloads = [
    {"text": f"passage about topic {i}", "language": "hi",
     "query_id": i, "passage_idx": 0, "chunk_idx": 0,
     "strategy": "smoke_test", "query_type": "DESCRIPTION",
     "is_selected": i % 2, "source": "english"}
    for i in range(10)
]
n = upsert_points(COLL, vecs, payloads)
print(f"   Upserted {n} points")

# 3. Search
print("3. Searching...")
query_vec = vecs[0]  # search for the first vector — should return itself at rank 0
t0 = time.perf_counter()
results = search(COLL, query_vec, top_k=5)
ms = (time.perf_counter() - t0) * 1000
print(f"   Search took {ms:.1f}ms, got {len(results)} results")
print(f"   Top result score: {results[0]['score']:.4f}  (expect ~1.0)")
assert results[0]["score"] > 0.99, f"Top score should be ~1.0, got {results[0]['score']}"

# 4. Collection info
print("4. Collection info...")
info = collection_info(COLL)
print(f"   points_count={info.get('points_count')}  vectors_count={info.get('vectors_count')}")

# 5. Language filter
print("5. Language filter (hi only)...")
results_hi = search(COLL, query_vec, top_k=10, language_filter="hi")
print(f"   Results with hi filter: {len(results_hi)}")

# Cleanup
print("6. Cleanup...")
get_client().delete_collection(COLL)
print("   Collection deleted")

print("\nQdrant smoke test PASSED")
