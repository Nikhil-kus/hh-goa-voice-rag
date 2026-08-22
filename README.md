# Voice-Enabled RAG System
### HH Goa 2026 — Shortlisting Task 2

Voice input → Sarvam STT → Multilingual RAG over MSMARCO-XI → Grounded answer

---

## Architecture

```
Browser (Next.js)
  → hold-to-record (MediaRecorder/WebRTC)
  → POST /api/transcribe  → Sarvam saaras:v3 STT
  → POST /api/query       → embed → Qdrant ANN → evidence gate → LLM → grounding check
  ← grounded answer | structured refusal + sources + per-stage latency
```

### Pipeline stages (every stage instrumented)
1. Audio validation
2. **Sarvam saaras:v3** STT (external — latency reported separately)
3. Query normalization (unicode NFC, whitespace)
4. Pre-query guardrails (unsafe content, off-topic vs corpus centroid)
5. Multilingual embedding (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim)
6. **Qdrant** ANN search with language filter
7. Evidence scoring gate (cosine similarity threshold)
8. LLM generation (modular — Groq / OpenAI)
9. Grounding verification (ROUGE-1 recall)
10. Structured response

---

## Dataset

**Source:** [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
— MS MARCO translated into 14 Indic languages.

**Indexed subset:**
| Language | Records | Passages | Source |
|----------|---------|----------|--------|
| Hindi (hi) | 1,000 | ~9,988 | val split |
| Bengali (bn) | 1,000 | ~9,988 | val split |
| Tamil (ta) | 1,000 | ~9,988 | val split |
| Kannada (kn) | 1,000 | ~9,988 | val split |

**Why validation split:** Train parquets are ~3.5GB each; val parquets are ~440MB.
Architecture scales to full train data by increasing `RECORDS_PER_LANGUAGE`.

**`is_selected` usage:** Ground-truth relevance stored as Qdrant payload metadata
for offline Recall@K evaluation **only**. Never used in live retrieval scoring.

---

## Chunking Strategies (3)

| Strategy | Boundaries | Chunk size |
|----------|-----------|------------|
| `fixed_size_overlap` | Every 256 tokens, 32 overlap | Fixed ~256 tokens |
| `sentence_aware` | Sentence boundaries, ≤200 tokens | Variable |
| `passage_structure_aware` | Dataset passage boundaries + metadata | Variable (full passage) |

Each strategy is stored in a separate Qdrant collection. Selectable live in UI.

---

## Guardrails

| Check | When | Action |
|-------|------|--------|
| Empty transcript | Pre-query | Refusal |
| Unsafe content | Pre-query | Refusal |
| Off-topic (cosine vs centroid) | Pre-query | Refusal |
| Insufficient evidence (score < threshold) | Post-retrieval | Refusal (no LLM call) |
| Grounding verification (ROUGE-1) | Post-generation | Reject answer → Refusal |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| STT | Sarvam saaras:v3 REST API |
| Embedding | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector DB | Qdrant (local or server) |
| LLM | Groq / OpenAI (modular, env-configurable) |
| Backend | Python 3.11 + FastAPI |
| Frontend | Next.js 15 + TypeScript + Tailwind CSS |
| Deployment | Docker Compose |

---

## Setup

### 1. Clone and configure

```bash
git clone <repo>
cd voice-rag-system
cp backend/.env.example backend/.env
# Fill in SARVAM_API_KEY, GROQ_API_KEY (or OPENAI_API_KEY)
```

### 2. Download data

```bash
cd backend
pip install -r requirements.txt
python scripts/download_data.py   # downloads ~1.8GB val parquets
```

### 3. Build the index (offline, one-time)

```bash
# Adjust RECORDS_PER_LANGUAGE in .env (default 1000, max ~6500 for val split)
python scripts/ingest.py
```

### 4. Prepare evaluation set

```bash
python scripts/prepare_eval.py
```

### 5. Run locally

```bash
# Backend
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install && npm run dev
```

Open http://localhost:3000

---

## Docker (local full stack)

```bash
docker compose up --build
```

The Qdrant data directory (`backend/qdrant_data/`) is mounted as a volume.
Pre-index locally (`python scripts/ingest.py`), then run the stack.

---

## Production Deployment (Railway + Qdrant Cloud + Vercel)

### Step 1 — Qdrant Cloud (free tier)
1. Sign up at https://cloud.qdrant.io
2. Create a free cluster (1GB)
3. Note the cluster URL and API key

### Step 2 — Build the index against Qdrant Cloud
```bash
# Set these in backend/.env
USE_QDRANT_LOCAL=false
QDRANT_HOST=your-cluster.qdrant.io
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key

# Run ingestion (uploads to cloud)
python scripts/download_data.py
python scripts/ingest.py
```

### Step 3 — Deploy backend to Railway
1. Go to https://railway.app → New Project → Deploy from GitHub
2. Select `hh-goa-voice-rag` repo, set root directory to `backend/`
3. Add environment variables (all from `.env`, with Qdrant Cloud values)
4. Railway auto-detects the Dockerfile and deploys
5. Note the Railway backend URL (e.g. `https://voice-rag-backend.up.railway.app`)

### Step 4 — Deploy frontend to Vercel
1. Go to https://vercel.com → New Project → Import from GitHub
2. Select `hh-goa-voice-rag`, set root directory to `frontend/`
3. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-railway-backend-url`
4. Deploy

### Step 5 — Verify
```
GET https://your-railway-backend-url/health
# Should return: {"status": "ok", "qdrant": "ok", ...}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SARVAM_API_KEY` | Yes | — | Sarvam AI subscription key |
| `LLM_PROVIDER` | Yes | groq | `groq` \| `openai` |
| `GROQ_API_KEY` | If groq | — | Groq API key |
| `OPENAI_API_KEY` | If openai | — | OpenAI API key |
| `LLM_MODEL` | No | llama-3.1-8b-instant | Model name for provider |
| `USE_QDRANT_LOCAL` | No | true | `false` for Docker/server mode |
| `QDRANT_HOST` | If server | localhost | Qdrant server host |
| `QDRANT_LOCAL_PATH` | If local | ./qdrant_data | Local storage path |
| `RECORDS_PER_LANGUAGE` | No | 1000 | Records to index per language |
| `LANGUAGES_TO_INDEX` | No | hi,bn,ta,kn | Comma-separated ISO 639-1 codes |
| `EVIDENCE_THRESHOLD` | No | 0.35 | Min cosine score for generation |
| `GROUNDING_THRESHOLD` | No | 0.15 | Min ROUGE-1 for answer acceptance |
| `RERANKER_ENABLED` | No | false | Enable dot-product reranker |

---

## Running Benchmarks

```bash
cd backend

# Latency benchmark (P50/P70/P100 per stage)
python scripts/benchmark_latency.py

# Retrieval quality (Recall@1/3/5/10 using is_selected)
python scripts/benchmark_retrieval.py

# End-to-end text query smoke test
python scripts/smoke_test_query.py
```

Results written to `benchmark_latency_{ts}.json/txt` and `benchmark_retrieval_{ts}.json/txt`.

---

## Tests

```bash
cd backend
python -m pytest tests/ -v
# 89 tests: chunker (34), guardrails (31), retriever (24)
```

---

## Limitations

- Index covers validation splits only (train files too large for download environment)
- `RECORDS_PER_LANGUAGE=1000` represents ~15% of each language's val split
- Telugu (te) excluded from index: no train parquet exists in dataset
- Full 14-language, 11.45M-record coverage requires larger compute

## Scaling

To index more data: increase `RECORDS_PER_LANGUAGE` in `.env` and re-run `ingest.py`.
To add languages: add to `LANGUAGES_TO_INDEX` (must be Sarvam STT supported).
To use train splits: requires ~3.5GB/language download; change `LANG_TO_VAL_PREFIX`
to `LANG_TO_FILE_PREFIX` in `ingest.py`.
