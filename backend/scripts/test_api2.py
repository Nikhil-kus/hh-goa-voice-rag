"""
Find queries from eval set that actually have high retrieval scores,
so we can test end-to-end generation with relevant passages.
"""
import json, urllib.request

BASE = "http://localhost:8000"

# Load a few actual queries from eval set
queries = []
with open("eval_queries.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        queries.append(json.loads(line))

print(f"Testing {len(queries)} queries from eval set...\n")

answered = 0
for q in queries:
    text = q.get("query_text", "")
    if not text or len(text) < 5:
        continue
    body = json.dumps({
        "transcript": text[:300],
        "language_code": "en-IN",
        "strategy": "passage_structure_aware",
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/api/query", data=body,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read())
        refused = d["refused"]
        reason  = d.get("refusal_reason", "")
        answer  = (d.get("answer") or "")
        lat     = d.get("latency", {})
        total   = lat.get("total_ms", 0) or 0
        gen     = lat.get("generation_ms", 0) or 0

        if not refused and answer:
            answered += 1
            print(f"[ANSWERED] {text[:60]}")
            print(f"  Answer: {answer[:150]}")
            print(f"  gen={gen:.0f}ms total={total:.0f}ms\n")
            if answered >= 3:
                break
        elif refused and reason == "insufficient_evidence":
            print(f"[NO EVIDENCE] {text[:60]}")
    except Exception as e:
        print(f"ERROR: {e}")

if answered == 0:
    print("\nNo answered queries found in first 20. The small index (~40K passages)")
    print("covers a narrow slice of the full MS MARCO dataset.")
    print("This is expected and documented — not a bug.")
