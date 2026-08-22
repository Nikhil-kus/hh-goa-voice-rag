import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, ".")

from app.services.embedder import embed_query
from app.services.vector_store import search, all_collection_info
from app.services.retriever import normalize_query
from app.config import settings

print("Collections:")
info = all_collection_info()
for name, d in info.items():
    print(f"  {name}: points={d.get('points_count')} status={d.get('status','?')}")

q = normalize_query("What was the Manhattan Project")
v = embed_query(q)
print(f"\nQuery: '{q}'")
print(f"Vector norm: {float((v**2).sum()**0.5):.4f}")

print("\nSearch (no language filter):")
results = search("msmarco_xi_passage_structure_aware", v, top_k=3, language_filter=None)
print(f"  {len(results)} results")
for r in results[:2]:
    score = r["score"]
    text = r["payload"]["text"][:80]
    lang = r["payload"]["language"]
    print(f"  score={score:.4f} lang={lang} text={text}")

print("\nSearch (hi filter):")
results2 = search("msmarco_xi_passage_structure_aware", v, top_k=3, language_filter="hi")
print(f"  {len(results2)} results")
for r in results2[:2]:
    score = r["score"]
    text = r["payload"]["text"][:80]
    print(f"  score={score:.4f} text={text}")
