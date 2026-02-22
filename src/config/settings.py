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
# If best retrieval score is below this, return "I don't have enough information" (no LLM call)
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

# API: max length for question (characters); requests exceeding this return 400
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))

# Rate limiting for /ask (per-IP). Set to 0 to disable.
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "0"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Generation (OpenAI or OpenAI-compatible: Ollama, Groq, OpenRouter)
# Override with LLM_MODEL in .env; use OPENAI_BASE_URL for non-OpenAI. See .env.example.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
