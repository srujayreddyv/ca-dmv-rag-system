# Setup and Process

This document outlines how to set up the development environment and the end-to-end process for the CA DMV RAG system.

---

## 1. Environment Setup

### Prerequisites

- **Python 3.10+** (3.11 recommended; CI and Docker use 3.11)
- **pip**

### Steps

1. **Clone the repository** (if needed)
   ```bash
   git clone https://github.com/srujayreddyv/ca-dmv-rag-system.git
   cd ca-dmv-rag-system
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables** (for the LLM and optionally Streamlit)
   - Copy `cp .env.example .env` and choose one LLM option:
   - **OpenAI (paid):** `OPENAI_API_KEY=sk-...`
   - **Ollama (free, local):** [Install Ollama](https://ollama.com), run `ollama pull llama3.2`, then `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_API_KEY=ollama`, `LLM_MODEL=llama3.2`
   - **Groq (free tier):** `OPENAI_BASE_URL=https://api.groq.com/openai/v1`, `OPENAI_API_KEY=gsk_...`, `LLM_MODEL=llama-3.1-70b-versatile` — get a key at [console.groq.com](https://console.groq.com/keys)
   - **OpenRouter (some free models):** `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, `OPENAI_API_KEY=sk-or-...`, `LLM_MODEL=google/gemma-2-9b-it:free` — get a key at [openrouter.ai/keys](https://openrouter.ai/keys)
   - **Streamlit:** optional `API_URL` to override the API base (default `http://localhost:8000`).

---

## 2. RAG Process Overview

The system follows a standard Retrieval Augmented Generation pipeline over the California DMV Driver Handbook.

### High-level flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   INGESTION     │     │   RETRIEVAL     │     │   GENERATION    │
│                 │     │                 │     │                 │
│  PDF → extract  │ ──► │  embed & index  │ ──► │  prompt + LLM   │
│  → clean → chunk│     │  → query →      │     │  → answer       │
│                 │     │  top-k chunks   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 3. Process Stages (Detailed)

### 3.1 Ingestion

**Goal:** Turn the raw handbook PDF into structured text suitable for embedding and search.

| Step        | Description | Output / Artifact      |
|------------|-------------|------------------------|
| Load PDF   | Extract text from `data/ca-drivers-handbook.pdf` using a PDF library (e.g. pypdf). | Raw text string(s) |
| Clean text | Normalize whitespace, fix encoding, remove headers/footers or other artifacts. | Cleaned text      |
| Chunk      | Split text into overlapping chunks (e.g. by character or token count). | `chunks` (list of `{text, metadata}`) |

**Design choices:**
- Chunk size and overlap affect recall and context. Typical: 256–512 tokens with 50–100 token overlap.
- Metadata (page, section) helps with citations and debugging.

---

### 3.2 Retrieval

**Goal:** Build a searchable index and, at query time, fetch the most relevant chunks.

| Step        | Description | Output / Artifact      |
|------------|-------------|------------------------|
| Embed      | Compute vector embeddings for each chunk (e.g. sentence-transformers or OpenAI). | Embedding vectors |
| Index      | Store embeddings in a vector store (e.g. FAISS) and persist to disk. | `handbook.index` (or similar) |
| Query      | For a user question: embed the query, run similarity search, return top-k chunks. | Top-k chunks with scores |

**Design choices:**
- Embedding model: local (e.g. `all-MiniLM-L6-v2`) vs. API (OpenAI).
- `top_k`: often 3–5 for QA; tune based on context window and accuracy.

---

### 3.3 Generation

**Goal:** Use an LLM to produce an answer grounded in the retrieved chunks.

| Step        | Description | Output / Artifact      |
|------------|-------------|------------------------|
| Prompt     | Build a prompt with: system/role, retrieved context (chunks), and the user question. | Prompt string     |
| Generate   | Call the LLM with the prompt; optionally set temperature, max_tokens. | Model response    |
| Postprocess| Strip boilerplate, add citations if desired. | Final answer      |

**Design choices:**
- Instruct the model to answer only from the context and say “I don’t know” when the context is insufficient.
- Include chunk metadata in the prompt if you want page/section references in the answer.

---

## 4. Status

| Component | Status |
|-----------|--------|
| **Ingestion** | Done — PDF load, clean, chunk → `chunks.jsonl` |
| **Retrieval** | Done — sentence-transformers + FAISS, `build_index`, `query_demo` |
| **Generation** | Done — prompts, `answer()`, OpenAI + Ollama / Groq / OpenRouter |
| **Ask flow** | Done — `ask.py`, `run_build_and_ask.sh` |
| **Evaluation** | Done — `data/eval/questions.json`, `scripts/run_eval.py`, exact match / ref-in-pred / LLM-judge |
| **Tests** | Done — `pytest tests/`, ingestion / evaluation / generation / retrieval (VectorStore); `-m "not slow"` to skip embed |
| **API** | Done — FastAPI `GET /`, `/docs` (Swagger), `/redoc`, `/openapi.json`, `GET /health`, `POST /ask`, `POST /retrieve` |
| **Streamlit** | Done — `streamlit run streamlit_app.py`; calls `POST /ask` |

---

## 5. Data and Artifacts

| Path / artifact | Role |
|-----------------|------|
| `data/ca-drivers-handbook.pdf` | Raw source PDF |
| `data/processed/chunks.jsonl` | Chunks from `build_chunks` |
| `data/embeddings/handbook.index` | FAISS index from `build_index` (in `.gitignore`) |
| `data/embeddings/handbook.meta.json` | Chunk metadata for the index |
| `data/eval/questions.json` | Gold Q&A for evaluation (10–20 handbook questions + reference answers) |

---

## 6. Running the Project

- **1) Build chunks (ingestion; run when the PDF changes):**
  ```bash
  python scripts/build_chunks.py
  ```

- **2) Build the vector index (retrieval; run when chunks change):**
  ```bash
  python scripts/build_index.py
  ```

- **3) Try a retrieval query:**
  ```bash
  python scripts/query_demo.py
  python scripts/query_demo.py "What is the speed limit on a highway?"
  ```

- **4) Ask a question (RAG: retrieve + LLM).** Configure an LLM in `.env` (see `.env.example`):
  ```bash
  python scripts/ask.py
  python scripts/ask.py "What is the blood alcohol limit for DUI?"
  ```

- **Or build index + ask in one step** (after `build_chunks` has been run):
  ```bash
  ./scripts/run_build_and_ask.sh
  ./scripts/run_build_and_ask.sh "When must I use headlights?"
  ```

- **5) Run evaluation** (after chunks and index or chunks-only; needs LLM in `.env`):
  ```bash
  python scripts/run_eval.py
  python scripts/run_eval.py --llm-judge
  ```
  Uses `data/eval/questions.json`. Reports `exact_match` and `ref_in_pred`; `--llm-judge` adds LLM-as-judge. Use to tune chunk size, `top_k`, or the prompt.

- **Optional: Clean cache:** `./scripts/clean.sh` — removes `__pycache__`, `.DS_Store`, `.ipynb_checkpoints` (not `.venv`).

- **6) API and Streamlit:**
  ```bash
  ./scripts/run_api.sh              # or: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
  streamlit run streamlit_app.py    # in another terminal; set API_URL in .env to override http://localhost:8000
  ```
  - **GET /** — info and links to docs/endpoints.
  - **GET /docs** — Swagger UI. **GET /redoc** — ReDoc. **GET /openapi.json** — OpenAPI schema.
  - **GET /health** — 200 if index or chunks exist.
  - **POST /ask** — body `{"question": "..."}` → `{"answer": "..."}`. Streamlit uses this.
  - **POST /retrieve** — body `{"question": "...", "top_k": 5}` → `{"chunks": [{text, page, score}]}` (no LLM).
  - The API uses the persisted index (`data/embeddings/handbook.index`) when present; otherwise it builds an in-memory index from `chunks.jsonl` on first `/ask` or `/retrieve`.

- **7) Tests:**
  ```bash
  pytest tests/
  pytest tests/ -m "not slow"    # skip slow tests (embed model load)
  pytest tests/ -m "slow"       # only slow tests
  ```
  Covers ingestion, evaluation, generation, retrieval (VectorStore), and API (`/, /health, /ask, /retrieve` with mocks). Slow tests need sentence-transformers.

---

## 7. API Reference (summary)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Info, `status`, and links to `/docs`, `/redoc`, `/openapi.json` |
| GET | `/docs` | Swagger UI (interactive) |
| GET | `/redoc` | ReDoc |
| GET | `/openapi.json` | OpenAPI 3.0 schema |
| GET | `/health` | Readiness: 200 if index or chunks exist |
| POST | `/ask` | RAG: `{"question":"..."}` → `{"answer":"..."}` |
| POST | `/retrieve` | Retrieval only: `{"question":"...", "top_k":5}` → `{"chunks":[{text, page, score}]}` |

---

## 8. References

- [California DMV Driver Handbook](https://www.dmv.ca.gov/portal/handbook/california-driver-handbook/) (source for `ca-drivers-handbook.pdf`)
- RAG: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

