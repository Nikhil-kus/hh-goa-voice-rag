"""Direct test of the retrieve() function with a real Qdrant query."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, ".")

from app.models.schemas import ChunkingStrategy
from app.services.retriever import retrieve
from app.services.embedder import get_model
from app.services.vector_store import get_client
from app.utils.timing import StageTimer

# Pre-warm
get_model()
get_client()

timer = StageTimer()
sources, max_score = retrieve(
    query="What was the Manhattan Project?",
    strategy=ChunkingStrategy.passage_structure_aware,
    language_code="en-IN",
    top_k=10,
    reranker_enabled=False,
    timer=timer,
)
ms = timer.all_ms()
print(f"Sources: {len(sources)}  max_score: {max_score:.4f}")
print(f"Latency: embed={ms.get('embedding',0):.1f}ms  retrieval={ms.get('vector_retrieval',0):.1f}ms")
for s in sources[:3]:
    print(f"  score={s.score:.4f} lang={s.language} text={s.text[:80]}")
