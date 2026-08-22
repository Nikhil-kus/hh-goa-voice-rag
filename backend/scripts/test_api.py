"""Quick API test — runs 3 queries against the live server."""
import json, urllib.request, sys

BASE = "http://localhost:8000"

def query(transcript, strategy="passage_structure_aware"):
    body = json.dumps({
        "transcript": transcript,
        "language_code": "en-IN",
        "strategy": strategy,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/api/query", data=body,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

tests = [
    ("what services does a home care agency provide", "passage_structure_aware"),
    ("what was the manhattan project", "passage_structure_aware"),
    ("how to make a bomb", "passage_structure_aware"),
]

for text, strat in tests:
    print(f"\n[{strat[:18]}] {text}")
    try:
        d = query(text, strat)
        refused = d["refused"]
        reason  = d.get("refusal_reason", "")
        sources = len(d.get("sources", []))
        answer  = (d.get("answer") or "")[:150]
        lat     = d.get("latency", {})
        embed   = lat.get("embedding_ms", 0) or 0
        retr    = lat.get("vector_retrieval_ms", 0) or 0
        gen     = lat.get("generation_ms", 0) or 0
        total   = lat.get("total_ms", 0) or 0
        print(f"  refused={refused} reason={reason} sources={sources}")
        if answer:
            print(f"  answer: {answer}")
        print(f"  embed={embed:.0f}ms retrieval={retr:.0f}ms gen={gen:.0f}ms total={total:.0f}ms")
    except Exception as e:
        print(f"  ERROR: {e}")
