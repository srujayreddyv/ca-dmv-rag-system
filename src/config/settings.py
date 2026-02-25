"""Paths and constants for the RAG pipeline."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Single default PDF. For multiple PDFs set env PDF_PATHS (comma-separated paths under data/).
PDF_PATH = DATA_DIR / "ca-drivers-handbook.pdf"


def _pdf_paths() -> list[Path]:
    raw = os.getenv("PDF_PATHS", "").strip()
    if not raw:
        return [PDF_PATH]
    return [DATA_DIR / p.strip() for p in raw.split(",") if p.strip()]


PDF_PATHS = _pdf_paths()
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
INDEX_PATH = EMBEDDINGS_DIR / "handbook.index"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
# Use paragraph-aware chunking (chunk_by_paragraphs) when True
USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "").lower() in ("1", "true", "yes")

# Embedding model (sentence-transformers)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval
TOP_K = 5
# Reranker: retrieve more then rerank with cross-encoder. Set USE_RERANKER=1 to enable.
USE_RERANKER = os.getenv("USE_RERANKER", "").lower() in ("1", "true", "yes")
RERANK_RETRIEVE_K = int(os.getenv("RERANK_RETRIEVE_K", "15"))
# Lightweight hybrid rerank (semantic + lexical), recommended for better quality on low-memory hosts.
USE_HYBRID_RERANK = os.getenv("USE_HYBRID_RERANK", "true").lower() in ("1", "true", "yes")
HYBRID_RERANK_ALPHA = float(os.getenv("HYBRID_RERANK_ALPHA", "0.75"))
# If best retrieval score is below this, return "I don't have enough information" (no LLM call)
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))
# Startup/runtime memory controls for hosted environments:
# - PRELOAD_RETRIEVER_ON_STARTUP: eagerly load retriever at startup (can increase memory on boot)
# - ALLOW_IN_MEMORY_INDEX_FALLBACK: build a temporary in-memory index from chunks when FAISS index is missing
PRELOAD_RETRIEVER_ON_STARTUP = os.getenv("PRELOAD_RETRIEVER_ON_STARTUP", "false").lower() in ("1", "true", "yes")
ALLOW_IN_MEMORY_INDEX_FALLBACK = os.getenv("ALLOW_IN_MEMORY_INDEX_FALLBACK", "false").lower() in ("1", "true", "yes")

# API: max length for question (characters); requests exceeding this return 400
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))

# Rate limiting for /ask (per-IP). Set to 0 to disable.
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "0"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
# Rate-limit backend:
# - memory: per-process (simple dev mode)
# - sqlite: process-shared on the same host (recommended default)
RATE_LIMIT_BACKEND = os.getenv("RATE_LIMIT_BACKEND", "sqlite").strip().lower()
RATE_LIMIT_DB_PATH = Path(os.getenv("RATE_LIMIT_DB_PATH", str(PROJECT_ROOT / "data" / "rate_limit.db")))

# API CORS configuration.
# Examples:
#   CORS_ALLOW_ORIGINS=*  (dev default)
#   CORS_ALLOW_ORIGINS=https://app.example.com,https://admin.example.com
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() in ("1", "true", "yes")

# Generation (OpenAI or OpenAI-compatible: Ollama, Groq, OpenRouter)
# Override with LLM_MODEL in .env; use OPENAI_BASE_URL for non-OpenAI. See .env.example.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
