"""Quick smoke test for embedder — run standalone to avoid TF startup noise."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embedder import embed_query, embed_batch, get_model

print("Loading model...")
t0 = time.perf_counter()
get_model()
load_ms = (time.perf_counter() - t0) * 1000
print(f"  Model loaded in {load_ms:.0f}ms")

# Single query
queries = [
    "what is the manhattan project",
    "मैनहट्टन परियोजना क्या है",
    "মহাত্মা গান্ধী সম্পর্কে বলুন",
]
for q in queries:
    t0 = time.perf_counter()
    v = embed_query(q)
    ms = (time.perf_counter() - t0) * 1000
    norm = float((v**2).sum()**0.5)
    print(f"  [{ms:5.1f}ms] shape={v.shape} norm={norm:.4f}  '{q[:50]}'")

# Batch
print("\nBatch (16 texts)...")
texts = [f"sample text number {i} about science history geography" for i in range(16)]
t0 = time.perf_counter()
vecs = embed_batch(texts, batch_size=16)
ms = (time.perf_counter() - t0) * 1000
print(f"  Batch {len(texts)} texts in {ms:.0f}ms → {vecs.shape}")

print("\nEmbedder smoke test PASSED")
