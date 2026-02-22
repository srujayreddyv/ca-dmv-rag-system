# CA DMV RAG System

RAG over the **California DMV Driver Handbook**: ingestion → retrieval → generation. Ask questions and get answers grounded in the handbook.

## What’s implemented

- **Ingestion:** PDF → clean → chunk → `data/processed/chunks.jsonl`
- **Retrieval:** sentence-transformers + FAISS → `data/embeddings/handbook.index`
- **Generation:** RAG prompt + LLM (OpenAI or OpenAI-compatible: Ollama, Groq, OpenRouter)
- **Ask:** `scripts/ask.py` — retrieve top‑k chunks and generate an answer
- **Evaluation:** `data/eval/questions.json` + `scripts/run_eval.py` — exact match, ref-in-pred, optional LLM-judge
- **API:** FastAPI `GET /docs`, `GET /health`, `POST /ask`, `POST /ask/stream` (SSE), `POST /retrieve` — `./scripts/run_api.sh`
- **Streamlit:** `streamlit run streamlit_app.py` — UI with optional **streaming** answers; session history; “Clear history” in sidebar
- **Docker:** `Dockerfile` + `docker-compose.yml` — `docker compose up` runs API (8000) + Streamlit (8501); mount `./data` and use `.env`
- **Logging:** `/ask` logs question length, confidence, latency, and errors (INFO/WARNING)
- **Deploy:** `render.yaml` Blueprint + [docs/DEPLOY.md](docs/DEPLOY.md) for Render (and other hosts)
- **Rate limiting:** Optional per-IP limit for `/ask` and `/ask/stream` via `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` (0 = off)
- **Reranker:** Optional cross-encoder rerank (`USE_RERANKER=1`, `RERANK_RETRIEVE_K`); better relevance, more latency
- **Eval:** `run_eval.py` — exact match, ref-in-pred, **embedding_similarity**, **BLEU**, **ROUGE-L F1**, optional `--llm-judge`; results in `data/eval/results/`
- **CI:** `.github/workflows/ci.yml` — test job (pytest) on every push/PR; **eval** job on push to main (needs `OPENAI_API_KEY` secret; see [DEPLOY.md](docs/DEPLOY.md))
- **Incremental index:** `scripts/append_index.py <new-chunks.jsonl>` — append new chunks to existing index without full rebuild
- **API options:** `include_sources` (bool), `source_filter` (optional) on `/ask`, `/ask/stream`, `/retrieve`
- **Multiple PDFs:** Set `PDF_PATHS` (comma-separated under `data/`); chunks get `source`; filter with `source_filter`
- **Semantic chunking:** `USE_SEMANTIC_CHUNKING=1` for paragraph-aware chunking
- **Streamlit:** Sidebar options **Include sources**, **Source filter** (for multiple PDFs)
- **Tests:** `pytest tests/` — ingestion, evaluation, generation, retrieval (VectorStore), API (/, /health, /ask, /retrieve); `-m "not slow"` to skip embed tests

## Requirements

- **Python 3.10+** (3.11 recommended). CI and Docker use 3.11.

## Quick start

```bash
git clone https://github.com/srujayreddyv/ca-dmv-rag-system.git
cd ca-dmv-rag-system

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.lock

cp .env.example .env
# Edit .env: choose OpenAI, or Ollama / Groq / OpenRouter (see .env.example)
```

**Ollama (free, local):** Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, then in `.env`:

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
LLM_MODEL=llama3.2
```

**Build and ask:**

```bash
python scripts/build_chunks.py      # if not done: PDF → chunks
python scripts/build_index.py       # chunks → vector index
python scripts/ask.py "What is the blood alcohol limit for DUI?"
```

Or in one step (after chunks exist):

```bash
./scripts/run_build_and_ask.sh "What is the blood alcohol limit for DUI?"
```

**API + Streamlit (after chunks/index and .env):**

```bash
./scripts/run_api.sh                # terminal 1: API at http://localhost:8000
streamlit run streamlit_app.py     # terminal 2: UI at http://localhost:8501
```

**Docker (after `build_chunks` + `build_index` and `.env`):**

```bash
docker compose up
# API http://localhost:8000  |  Streamlit http://localhost:8501
# Data and .env are mounted from the host.
```

## Project layout

```
ca-dmv-rag-system/
├── data/
│   ├── ca-drivers-handbook.pdf    # source PDF
│   ├── processed/chunks.jsonl     # from build_chunks
│   ├── eval/questions.json        # gold Q&A for evaluation
│   └── embeddings/                # handbook.index (from build_index); ignored in git
├── streamlit_app.py               # Streamlit UI (/ask + /ask/stream, history)
├── notebooks/                      # ingestion, retrieval, generation, evaluation inspection
├── tests/                         # pytest: ingestion, evaluation, generation, retrieval, API
├── pytest.ini
├── scripts/
│   ├── build_chunks.py            # PDF → chunks
│   ├── build_index.py            # chunks → FAISS index
│   ├── append_index.py           # append new chunks to existing index (incremental)
│   ├── query_demo.py             # retrieval-only CLI (no LLM)
│   ├── ask.py                    # RAG: retrieve + LLM
│   ├── run_eval.py               # evaluation; writes data/eval/results/
│   ├── run_api.sh                # uvicorn src.api.main:app
│   ├── run_build_and_ask.sh      # build index + ask
│   └── clean.sh                  # remove __pycache__, .DS_Store, .ipynb_checkpoints
├── src/
│   ├── api/main.py               # FastAPI: /docs, /health, /ask, /ask/stream, /retrieve
│   ├── config/settings.py
│   ├── ingestion/                # pdf_loader, clean_text, chunker
│   ├── retrieval/                # embed, vector_store, retriever
│   ├── generation/               # prompts, answer (+ answer_stream)
│   └── evaluation/               # metrics: exact_match, ref_in_pred, llm_judge
├── Dockerfile                    # API + Streamlit (same image)
├── docker-compose.yml            # api + streamlit; mount data/, .env
├── render.yaml                   # Render Blueprint (see docs/DEPLOY.md)
├── .env.example
└── docs/
    ├── SETUP_AND_PROCESS.md      # setup, RAG stages, run order
    ├── DEPLOY.md                 # deploy API (Render, etc.)
    └── ROADMAP.md                # done / next
```

## LLM options

| Provider   | Notes                         | .env.example section |
|-----------|-------------------------------|------------------------|
| **OpenAI** | Paid; needs billing           | Option A              |
| **Ollama** | Free, local; `ollama pull …`  | Option B              |
| **Groq**   | Free tier; fast               | Option C              |
| **OpenRouter** | Some free models          | Option D              |

See `.env.example` for the exact variables.

## Tests

```bash
pytest tests/
pytest tests/ -m "not slow"   # skip slow tests (embedding model)
```

## API endpoints (when running `./scripts/run_api.sh`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Info and links |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |
| GET | `/openapi.json` | OpenAPI schema |
| GET | `/health` | Readiness |
| GET | `/metrics` | Basic API metrics (requests, errors, latency, estimated token usage) |
| POST | `/ask` | RAG: `{"question":"..."}` → `{"answer","sources","confidence"}` |
| POST | `/ask/stream` | Same input; Server-Sent Events (tokens then `done` with answer/sources) |
| POST | `/retrieve` | Chunks only: `{"question":"...", "top_k":5}` → `{"chunks":[...]}` |

## Documentation

- [SETUP_AND_PROCESS.md](docs/SETUP_AND_PROCESS.md) — setup, RAG stages, run order
- [DEPLOY.md](docs/DEPLOY.md) — deploy API to Render (or other hosts)
- [ROADMAP.md](docs/ROADMAP.md) — what’s done and what to add next
- **GET /docs** (Swagger) when the API is running

## Next steps

- **Run locally:** Finish [Quick start](#quick-start), then `./scripts/run_api.sh` and `streamlit run streamlit_app.py`.
- **Deploy:** Push to GitHub, connect the repo in [Render](https://render.com), set env vars, deploy (see [DEPLOY.md](docs/DEPLOY.md)).
- **Improve RAG:** See [ROADMAP.md](docs/ROADMAP.md) for optional reranker, extra eval metrics, or multiple PDFs.

## License

See [LICENSE](LICENSE).
