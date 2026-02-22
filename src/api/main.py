"""FastAPI app: /docs (Swagger), /redoc, /health, /ask, /retrieve, /ask/stream."""

import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional

from pydantic import BaseModel, Field

from src.config.settings import (
    CHUNKS_JSONL,
    INDEX_PATH,
    MAX_QUESTION_LENGTH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    RERANK_RETRIEVE_K,
    SCORE_THRESHOLD,
    TOP_K,
    USE_RERANKER,
)
from src.generation import answer, answer_stream
from src.retrieval import Retriever, embed_query, embed_texts
from src.retrieval.reranker import rerank
from src.retrieval.vector_store import VectorStore

# In-memory rate limit: ip -> list of request timestamps (for /ask and /ask/stream)
_rate_limit_timestamps: dict[str, list[float]] = defaultdict(list)

app = FastAPI(
    title="CA DMV RAG API",
    description="""RAG over the California DMV Driver Handbook.

- **GET /docs** — Swagger UI (interactive)
- **GET /redoc** — ReDoc
- **GET /openapi.json** — OpenAPI schema
- **GET /health** — readiness (index or chunks exist)
- **POST /ask** — RAG: retrieve + LLM answer
- **POST /retrieve** — retrieval only (top-k chunks, no LLM)
""",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _client_ip(request: Request) -> str:
    """Prefer X-Forwarded-For when behind a proxy (e.g. Render)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request) -> None:
    """Raise 429 if client has exceeded rate limit for /ask. No-op if RATE_LIMIT_REQUESTS is 0."""
    if RATE_LIMIT_REQUESTS <= 0:
        return
    ip = _client_ip(request)
    now = time.monotonic()
    timestamps = _rate_limit_timestamps[ip]
    # Drop timestamps outside the window
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    timestamps.append(now)


_retriever = None


def _build_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    if INDEX_PATH.exists():
        _retriever = Retriever(INDEX_PATH)
        return _retriever
    if not CHUNKS_JSONL.exists():
        raise FileNotFoundError(
            f"Need {CHUNKS_JSONL} or {INDEX_PATH}. Run: python scripts/build_chunks.py [build_index.py]"
        )
    chunks_list = []
    with open(CHUNKS_JSONL) as f:
        for line in f:
            if line.strip():
                chunks_list.append(json.loads(line))
    texts = [c["text"] for c in chunks_list]
    embs = embed_texts(texts)
    store = VectorStore()
    store.add(embs, chunks_list)

    class _InMemoryRetriever:
        def retrieve(self, q, top_k=5):
            return store.search(embed_query(q), top_k=top_k)

    _retriever = _InMemoryRetriever()
    return _retriever


@app.get("/", tags=["Root"])
def root():
    """Links to docs and endpoints."""
    healthy = INDEX_PATH.exists() or CHUNKS_JSONL.exists()
    return {
        "name": "CA DMV RAG API",
        "version": "0.1.0",
        "status": "ok" if healthy else "degraded",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "endpoints": {
            "health": {"GET": "/health"},
            "ask": {"POST": "/ask"},
            "ask_stream": {"POST": "/ask/stream"},
            "retrieve": {"POST": "/retrieve"},
        },
    }


@app.get("/health", tags=["Health"])
def health():
    """Check index/chunks exist. LLM config is validated on /ask."""
    if INDEX_PATH.exists() or CHUNKS_JSONL.exists():
        return {"status": "ok"}
    raise HTTPException(
        status_code=503,
        detail=f"Neither {INDEX_PATH} nor {CHUNKS_JSONL} found. Run build_chunks.py and optionally build_index.py.",
    )


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        max_length=MAX_QUESTION_LENGTH,
        description=f"Question about the handbook (max {MAX_QUESTION_LENGTH} characters)",
    )
    include_sources: bool = Field(default=True, description="If False, response has empty sources list")
    source_filter: Optional[str] = Field(default=None, description="Only use chunks from this source (e.g. handbook name)")


class SourceOut(BaseModel):
    page: Optional[int] = None
    score: float = Field(..., description="Similarity score (higher = more relevant)")
    snippet: str = Field(..., description="Short excerpt from the chunk")


class AskResponse(BaseModel):
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved chunks")
    sources: list[SourceOut] = Field(default_factory=list, description="Chunks retrieved from the handbook (vector DB)")
    confidence: Optional[str] = Field(
        default="high",
        description="'high' = answer from retrieved chunks; 'low' = no LLM call, not enough relevant context",
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(req: AskRequest, request: Request):
    """Run RAG: retrieve top-k chunks from vector DB and generate an answer with the LLM."""
    from openai import AuthenticationError, NotFoundError, RateLimitError

    _check_rate_limit(request)
    start = time.perf_counter()
    q = (req.question or "").strip() or "What is in the handbook?"
    question_preview = (q[:80] + "…") if len(q) > 80 else q

    try:
        retriever = _build_retriever()
    except FileNotFoundError as e:
        logger.warning("ask build_retriever failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    retrieve_k = RERANK_RETRIEVE_K if USE_RERANKER else TOP_K
    chunks = retriever.retrieve(q, top_k=retrieve_k)
    if USE_RERANKER and len(chunks) > TOP_K:
        chunks = rerank(q, chunks, top_k=TOP_K)
    if req.source_filter is not None:
        chunks = [c for c in chunks if c.get("source") == req.source_filter]

    snippet_len = 200
    sources_out = [
        SourceOut(
            page=c.get("page"),
            score=float(c.get("score", 0.0)),
            snippet=(c.get("text", "")[:snippet_len] + "..." if len(c.get("text", "")) > snippet_len else c.get("text", "")),
        )
        for c in chunks
    ]
    sources = sources_out if req.include_sources else []

    # Low-confidence: if best retrieval score is below threshold, skip LLM and return a safe message
    best_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
    if chunks and best_score < SCORE_THRESHOLD:
        out = (
            "I don't have enough relevant information in the handbook to answer that. "
            "Try rephrasing or asking something more specific to the California DMV Driver Handbook."
        )
        latency = time.perf_counter() - start
        logger.info(
            "ask question_len=%d confidence=low latency_seconds=%.3f",
            len(q), latency, extra={"question_preview": question_preview},
        )
        return AskResponse(answer=out, sources=sources, confidence="low")

    try:
        out = answer(q, context_chunks=chunks)
    except ValueError as e:
        logger.warning("ask llm error: %s", e)
        raise HTTPException(status_code=503, detail=f"LLM not configured: {e}")
    except AuthenticationError as e:
        logger.warning("ask auth error: %s", e)
        raise HTTPException(status_code=401, detail="Invalid API key. Check .env")
    except NotFoundError as e:
        logger.warning("ask model not found: %s", e)
        raise HTTPException(status_code=502, detail="Model not found. Check LLM_MODEL in .env")
    except RateLimitError as e:
        logger.warning("ask rate limit: %s", e)
        raise HTTPException(status_code=429, detail="LLM rate limit exceeded")

    latency = time.perf_counter() - start
    logger.info(
        "ask question_len=%d confidence=high latency_seconds=%.3f",
        len(q), latency, extra={"question_preview": question_preview},
    )
    return AskResponse(answer=out, sources=sources, confidence="high")


def _sse_event(data: dict) -> str:
    """One SSE event line (data: {...})."""
    return f"data: {json.dumps(data)}\n\n"


@app.post("/ask/stream", tags=["RAG"])
def ask_stream(req: AskRequest, request: Request):
    """Stream answer tokens as Server-Sent Events. Events: data: {\"token\": \"...\"} then data: {\"done\": true, \"answer\", \"sources\", \"confidence\"}."""
    from openai import AuthenticationError, NotFoundError, RateLimitError

    _check_rate_limit(request)
    q = (req.question or "").strip() or "What is in the handbook?"

    def generate():
        try:
            retriever = _build_retriever()
        except FileNotFoundError as e:
            yield _sse_event({"error": str(e)})
            return
        retrieve_k = RERANK_RETRIEVE_K if USE_RERANKER else TOP_K
        chunks = retriever.retrieve(q, top_k=retrieve_k)
        if USE_RERANKER and len(chunks) > TOP_K:
            chunks = rerank(q, chunks, top_k=TOP_K)
        if req.source_filter is not None:
            chunks = [c for c in chunks if c.get("source") == req.source_filter]
        snippet_len = 200
        sources_list = [
            {"page": c.get("page"), "score": float(c.get("score", 0.0)), "snippet": (c.get("text", "")[:snippet_len] + "..." if len(c.get("text", "")) > snippet_len else c.get("text", ""))}
            for c in chunks
        ]
        sources = sources_list if req.include_sources else []
        best_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
        if chunks and best_score < SCORE_THRESHOLD:
            out = (
                "I don't have enough relevant information in the handbook to answer that. "
                "Try rephrasing or asking something more specific to the California DMV Driver Handbook."
            )
            yield _sse_event({"token": out})
            yield _sse_event({"done": True, "answer": out, "sources": sources, "confidence": "low"})
            return
        try:
            full = []
            for token in answer_stream(q, context_chunks=chunks):
                full.append(token)
                yield _sse_event({"token": token})
            out = "".join(full)
            yield _sse_event({"done": True, "answer": out, "sources": sources, "confidence": "high"})
        except (ValueError, AuthenticationError, NotFoundError, RateLimitError) as e:
            yield _sse_event({"error": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class RetrieveRequest(BaseModel):
    question: str = Field(
        ...,
        max_length=MAX_QUESTION_LENGTH,
        description=f"Query to search the handbook (max {MAX_QUESTION_LENGTH} characters)",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to return")
    source_filter: Optional[str] = Field(default=None, description="Only return chunks from this source")


class ChunkOut(BaseModel):
    text: str
    page: Optional[int] = None
    score: float


class RetrieveResponse(BaseModel):
    chunks: list[ChunkOut] = Field(..., description="Top-k chunks (text, page, score)")


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
def retrieve(req: RetrieveRequest):
    """Retrieve top-k chunks for a question. No LLM; use for inspection or custom pipelines."""
    try:
        retriever = _build_retriever()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    q = (req.question or "").strip() or "What is in the handbook?"
    raw = retriever.retrieve(q, top_k=req.top_k)
    if req.source_filter is not None:
        raw = [c for c in raw if c.get("source") == req.source_filter]
    chunks = [ChunkOut(text=c.get("text", ""), page=c.get("page"), score=c.get("score", 0.0)) for c in raw]
    return RetrieveResponse(chunks=chunks)
