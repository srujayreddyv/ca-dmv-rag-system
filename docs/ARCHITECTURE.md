# CA DMV RAG System Architecture

This diagram reflects the current production-ready design: ingestion + FAISS indexing, FastAPI serving, Streamlit client, evaluation workflow, and Render deployment/runtime controls.

```mermaid
flowchart LR
  %% =========================
  %% Clients
  %% =========================
  User["End User"]
  UI["Streamlit UI (streamlit_app.py)"]
  Swagger["Swagger / ReDoc"]

  User --> UI
  User --> Swagger

  %% =========================
  %% API Layer
  %% =========================
  subgraph API["FastAPI Service (src/api/main.py)"]
    direction TB
    Ask["POST /ask"]
    AskStream["POST /ask/stream (SSE)"]
    Retrieve["POST /retrieve"]
    Health["GET /health"]
    Metrics["GET /metrics"]

    Middleware["Middleware + Controls:
    - Request ID
    - Structured errors
    - Rate limiting (memory/sqlite)
    - CORS"]

    Confidence["Confidence Gate:
    - SCORE_THRESHOLD
    - low-confidence safe response"]
  end

  UI --> Ask
  UI --> AskStream
  Swagger --> Health
  Swagger --> Retrieve
  Swagger --> Metrics

  Ask --> Middleware
  AskStream --> Middleware
  Retrieve --> Middleware

  Ask --> Confidence
  AskStream --> Confidence

  %% =========================
  %% Retrieval Layer
  %% =========================
  subgraph Retrieval["Retrieval Layer"]
    direction TB
    Retriever["Retriever + VectorStore
    (src/retrieval/*)"]
    FAISS["FAISS Index
    data/embeddings/handbook.index"]
    Meta["Chunk Metadata
    handbook.meta.json"]
    Hybrid["Hybrid Rerank
    semantic + lexical"]
    Cross["Optional Cross-Encoder Rerank
    USE_RERANKER=1"]
  end

  Ask --> Retriever
  AskStream --> Retriever
  Retrieve --> Retriever
  Retriever <--> FAISS
  Retriever <--> Meta
  Retriever --> Hybrid
  Retriever --> Cross

  %% =========================
  %% Generation Layer
  %% =========================
  subgraph Generation["Generation Layer"]
    direction TB
    Prompt["Prompt Builder
    (src/generation/prompts.py)"]
    LLM["LLM Gateway
    OpenAI / compatible endpoint"]
    Answer["Final Answer + Sources + Confidence"]
  end

  Hybrid --> Prompt
  Cross --> Prompt
  Prompt --> LLM --> Answer
  Answer --> Ask
  Answer --> AskStream

  %% =========================
  %% Data Pipeline
  %% =========================
  subgraph DataPipeline["Offline Data Pipeline"]
    direction TB
    PDF["Source PDFs
    data/*.pdf"]
    BuildChunks["scripts/build_chunks.py
    PDF -> cleaned chunks"]
    Chunks["chunks.jsonl
    data/processed/chunks.jsonl"]
    BuildIndex["scripts/build_index.py
    chunks -> embeddings -> FAISS"]
  end

  PDF --> BuildChunks --> Chunks --> BuildIndex --> FAISS
  Chunks --> Meta

  %% =========================
  %% Evaluation + CI/CD
  %% =========================
  subgraph Quality["Evaluation + CI/CD"]
    direction TB
    EvalQ["data/eval/questions.json"]
    RunEval["scripts/run_eval.py
    EM / Ref-in-Pred / BLEU / ROUGE-L / cosine"]
    Results["data/eval/results/*.json"]
    CI["GitHub Actions
    test + eval jobs"]
  end

  EvalQ --> RunEval --> Results
  CI --> RunEval
  CI --> Ask

  %% =========================
  %% Hosting
  %% =========================
  subgraph Hosting["Deployment (Render)"]
    direction TB
    Docker["Docker Image
    requirements.lock + app code"]
    StartScript["scripts/render_start.sh
    builds index if missing"]
    Runtime["Uvicorn Runtime
    port $PORT"]
  end

  Docker --> StartScript --> Runtime --> API
```

## Request Lifecycle (Ask)

```mermaid
sequenceDiagram
  participant U as User
  participant S as Streamlit
  participant A as FastAPI
  participant R as Retriever/FAISS
  participant G as Prompt+LLM

  U->>S: Ask question
  S->>A: POST /ask or /ask/stream
  A->>A: Validate input + rate-limit + request_id
  A->>R: Retrieve top-k chunks
  R-->>A: Chunks + scores
  A->>A: Hybrid/Cross rerank + confidence gate
  alt Low confidence
    A-->>S: Safe "not enough info" response
  else High confidence
    A->>G: Build prompt with context
    G-->>A: Answer
    A-->>S: Answer + sources + confidence
  end
  S-->>U: Render response (streamed or non-streamed)
```
