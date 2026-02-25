"""FastAPI app: /docs (Swagger), /redoc, /health, /ask, /retrieve, /ask/stream."""

import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
import warnings
from collections import defaultdict
from contextlib import asynccontextmanager
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
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional

from pydantic import BaseModel, Field

from src.config.settings import (
    ALLOW_IN_MEMORY_INDEX_FALLBACK,
    CHUNKS_JSONL,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_ORIGINS,
    INDEX_PATH,
    HYBRID_RERANK_ALPHA,
    MAX_QUESTION_LENGTH,
    PRELOAD_RETRIEVER_ON_STARTUP,
    RATE_LIMIT_BACKEND,
    RATE_LIMIT_DB_PATH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    RERANK_RETRIEVE_K,
    SCORE_THRESHOLD,
    TOP_K,
    USE_HYBRID_RERANK,
    USE_RERANKER,
)
from src.generation import answer, answer_stream
from src.retrieval import Retriever, embed_query, embed_texts
from src.retrieval.reranker import hybrid_rerank, rerank
from src.retrieval.vector_store import VectorStore

# In-memory rate limit: ip -> list of request timestamps (for /ask and /ask/stream)
_rate_limit_timestamps: dict[str, list[float]] = defaultdict(list)
_metrics_lock = threading.Lock()
_metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "path": defaultdict(lambda: {"requests": 0, "errors": 0, "latency_ms_sum": 0.0}),
    "status": defaultdict(int),
    "tokens_estimated": {"prompt_total": 0, "completion_total": 0},
}


def _parse_cors_origins(raw: str) -> list[str]:
    txt = (raw or "").strip()
    if not txt:
        return ["*"]
    if txt == "*":
        return ["*"]
    return [x.strip() for x in txt.split(",") if x.strip()]


_cors_origins = _parse_cors_origins(CORS_ALLOW_ORIGINS)
_allow_credentials = CORS_ALLOW_CREDENTIALS and _cors_origins != ["*"]


def _client_ip(request: Request) -> str:
    """Prefer X-Forwarded-For when behind a proxy (e.g. Render)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _init_rate_limit_db() -> None:
    if RATE_LIMIT_BACKEND != "sqlite":
        return
    RATE_LIMIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(RATE_LIMIT_DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS rate_limit (ip TEXT NOT NULL, ts REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limit_ip_ts ON rate_limit(ip, ts)")
        conn.commit()


def _estimate_tokens(text: str) -> int:
    # Fast heuristic for lightweight API metrics.
    return max(0, (len(text or "") + 3) // 4)


def _record_token_usage(prompt_text: str, completion_text: str) -> None:
    with _metrics_lock:
        _metrics["tokens_estimated"]["prompt_total"] += _estimate_tokens(prompt_text)
        _metrics["tokens_estimated"]["completion_total"] += _estimate_tokens(completion_text)


def _check_rate_limit(request: Request) -> None:
    """Raise 429 if client has exceeded rate limit for /ask. No-op if RATE_LIMIT_REQUESTS is 0."""
    if RATE_LIMIT_REQUESTS <= 0:
        return
    ip = _client_ip(request)
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    if RATE_LIMIT_BACKEND == "sqlite":
        try:
            with sqlite3.connect(RATE_LIMIT_DB_PATH, timeout=5.0) as conn:
                conn.execute("DELETE FROM rate_limit WHERE ts < ?", (cutoff,))
                row = conn.execute("SELECT COUNT(*) FROM rate_limit WHERE ip = ?", (ip,)).fetchone()
                count = int(row[0]) if row else 0
                if count >= RATE_LIMIT_REQUESTS:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
                    )
                conn.execute("INSERT INTO rate_limit (ip, ts) VALUES (?, ?)", (ip, now))
                conn.commit()
            return
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("rate limit sqlite backend failed (fallback=memory): %s", e)
    # Fallback memory backend
    timestamps = _rate_limit_timestamps[ip]
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    timestamps.append(now)


_retriever = None


def _llm_env_configured() -> bool:
    """True when an OpenAI-style API configuration is present."""
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    return bool(base_url or api_key)


def _build_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    if INDEX_PATH.exists():
        _retriever = Retriever(INDEX_PATH)
        return _retriever
    if not ALLOW_IN_MEMORY_INDEX_FALLBACK:
        raise FileNotFoundError(
            f"Index not found: {INDEX_PATH}. Run: python scripts/build_index.py"
        )
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


def startup_preflight() -> None:
    """
    Startup checks for MVP operability.
    Logs actionable warnings for missing artifacts/config and validates retriever load path.
    """
    index_exists = INDEX_PATH.exists()
    chunks_exists = CHUNKS_JSONL.exists()
    llm_ok = _llm_env_configured()
    logger.info("startup preflight: rate_limit_backend=%s", RATE_LIMIT_BACKEND)
    logger.info(
        "startup preflight: preload_retriever=%s in_memory_index_fallback=%s",
        PRELOAD_RETRIEVER_ON_STARTUP,
        ALLOW_IN_MEMORY_INDEX_FALLBACK,
    )
    if RATE_LIMIT_BACKEND == "sqlite":
        try:
            _init_rate_limit_db()
            logger.info("startup preflight: rate-limit sqlite db ready at %s", RATE_LIMIT_DB_PATH)
        except Exception as e:
            logger.warning("startup preflight: failed to init rate-limit db: %s", e)

    if index_exists:
        logger.info("startup preflight: index found at %s", INDEX_PATH)
    elif chunks_exists:
        if ALLOW_IN_MEMORY_INDEX_FALLBACK:
            logger.warning(
                "startup preflight: index missing (%s); API may build in-memory index from %s on request.",
                INDEX_PATH,
                CHUNKS_JSONL,
            )
        else:
            logger.warning(
                "startup preflight: index missing (%s); fallback disabled. Run build_index before serving traffic.",
                INDEX_PATH,
            )
    else:
        logger.warning(
            "startup preflight: missing both %s and %s. Run build scripts before handling traffic.",
            INDEX_PATH,
            CHUNKS_JSONL,
        )

    if llm_ok:
        logger.info("startup preflight: LLM env appears configured")
    else:
        logger.warning("startup preflight: LLM env not configured (OPENAI_API_KEY or OPENAI_BASE_URL)")

    if PRELOAD_RETRIEVER_ON_STARTUP and (index_exists or chunks_exists):
        try:
            _build_retriever()
            logger.info("startup preflight: retriever initialized")
        except Exception as e:
            logger.warning("startup preflight: retriever initialization failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_preflight()
    yield


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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_and_metrics(request: Request, call_next):
    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Let exception handlers format the response; still count as request/error.
        with _metrics_lock:
            _metrics["requests_total"] += 1
            _metrics["errors_total"] += 1
            p = _metrics["path"][request.url.path]
            p["requests"] += 1
            p["errors"] += 1
            p["latency_ms_sum"] += (time.perf_counter() - start) * 1000.0
            _metrics["status"]["500"] += 1
        raise
    latency_ms = (time.perf_counter() - start) * 1000.0
    status_key = str(response.status_code)
    with _metrics_lock:
        _metrics["requests_total"] += 1
        p = _metrics["path"][request.url.path]
        p["requests"] += 1
        p["latency_ms_sum"] += latency_ms
        _metrics["status"][status_key] += 1
        if response.status_code >= 400:
            _metrics["errors_total"] += 1
            p["errors"] += 1
    response.headers["X-Request-ID"] = _request_id(request)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        message = detail.get("message", str(detail))
        code = detail.get("code", f"http_{exc.status_code}")
    else:
        message = str(detail)
        code = f"http_{exc.status_code}"
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": message, "code": code, "request_id": _request_id(request)},
        headers={"X-Request-ID": _request_id(request), "X-Error-Code": code},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    code = "internal_error"
    logger.exception("unhandled error request_id=%s path=%s err=%s", _request_id(request), request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": code, "request_id": _request_id(request)},
        headers={"X-Request-ID": _request_id(request), "X-Error-Code": code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    code = "validation_error"
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "code": code, "request_id": _request_id(request)},
        headers={"X-Request-ID": _request_id(request), "X-Error-Code": code},
    )


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
            "metrics": {"GET": "/metrics"},
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
        detail={
            "code": "missing_data_artifacts",
            "message": f"Neither {INDEX_PATH} nor {CHUNKS_JSONL} found. Run build_chunks.py and optionally build_index.py.",
        },
    )


@app.get("/metrics", tags=["Health"])
def metrics():
    """Simple JSON metrics for API activity and estimated token usage."""
    with _metrics_lock:
        path_out = {}
        for path, v in _metrics["path"].items():
            avg = (v["latency_ms_sum"] / v["requests"]) if v["requests"] else 0.0
            path_out[path] = {
                "requests": v["requests"],
                "errors": v["errors"],
                "latency_ms_avg": round(avg, 3),
            }
        return {
            "requests_total": _metrics["requests_total"],
            "errors_total": _metrics["errors_total"],
            "status": dict(_metrics["status"]),
            "path": path_out,
            "tokens_estimated": dict(_metrics["tokens_estimated"]),
        }


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
        logger.warning("ask build_retriever failed request_id=%s err=%s", _request_id(request), e)
        raise HTTPException(status_code=503, detail={"code": "retriever_unavailable", "message": str(e)})

    retrieve_k = RERANK_RETRIEVE_K if USE_RERANKER else TOP_K
    chunks = retriever.retrieve(q, top_k=retrieve_k)
    if USE_RERANKER and len(chunks) > TOP_K:
        chunks = rerank(q, chunks, top_k=TOP_K)
    elif USE_HYBRID_RERANK:
        chunks = hybrid_rerank(q, chunks, top_k=TOP_K, alpha=HYBRID_RERANK_ALPHA)
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
            "ask request_id=%s question_len=%d confidence=low latency_seconds=%.3f",
            _request_id(request), len(q), latency, extra={"question_preview": question_preview},
        )
        prompt_text = q + "\n" + "\n".join(c.get("text", "") for c in chunks)
        _record_token_usage(prompt_text, out)
        return AskResponse(answer=out, sources=sources, confidence="low")

    try:
        out = answer(q, context_chunks=chunks)
    except ValueError as e:
        logger.warning("ask llm error request_id=%s err=%s", _request_id(request), e)
        raise HTTPException(status_code=503, detail={"code": "llm_not_configured", "message": f"LLM not configured: {e}"})
    except AuthenticationError as e:
        logger.warning("ask auth error request_id=%s err=%s", _request_id(request), e)
        raise HTTPException(status_code=401, detail={"code": "llm_auth_error", "message": "Invalid API key. Check .env"})
    except NotFoundError as e:
        logger.warning("ask model not found request_id=%s err=%s", _request_id(request), e)
        raise HTTPException(status_code=502, detail={"code": "llm_model_not_found", "message": "Model not found. Check LLM_MODEL in .env"})
    except RateLimitError as e:
        logger.warning("ask llm rate limit request_id=%s err=%s", _request_id(request), e)
        raise HTTPException(status_code=429, detail={"code": "llm_rate_limited", "message": "LLM rate limit exceeded"})

    latency = time.perf_counter() - start
    logger.info(
        "ask request_id=%s question_len=%d confidence=high latency_seconds=%.3f",
        _request_id(request), len(q), latency, extra={"question_preview": question_preview},
    )
    prompt_text = q + "\n" + "\n".join(c.get("text", "") for c in chunks)
    _record_token_usage(prompt_text, out)
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
            yield _sse_event({"error": str(e), "code": "retriever_unavailable", "request_id": _request_id(request)})
            return
        retrieve_k = RERANK_RETRIEVE_K if USE_RERANKER else TOP_K
        chunks = retriever.retrieve(q, top_k=retrieve_k)
        if USE_RERANKER and len(chunks) > TOP_K:
            chunks = rerank(q, chunks, top_k=TOP_K)
        elif USE_HYBRID_RERANK:
            chunks = hybrid_rerank(q, chunks, top_k=TOP_K, alpha=HYBRID_RERANK_ALPHA)
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
            yield _sse_event({"error": str(e), "code": "llm_error", "request_id": _request_id(request)})

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
        raise HTTPException(status_code=503, detail={"code": "retriever_unavailable", "message": str(e)})

    q = (req.question or "").strip() or "What is in the handbook?"
    retrieve_k = max(req.top_k, RERANK_RETRIEVE_K) if USE_RERANKER else req.top_k
    raw = retriever.retrieve(q, top_k=retrieve_k)
    if USE_RERANKER and len(raw) > req.top_k:
        raw = rerank(q, raw, top_k=req.top_k)
    elif USE_HYBRID_RERANK:
        raw = hybrid_rerank(q, raw, top_k=req.top_k, alpha=HYBRID_RERANK_ALPHA)
    if req.source_filter is not None:
        raw = [c for c in raw if c.get("source") == req.source_filter]
    chunks = [ChunkOut(text=c.get("text", ""), page=c.get("page"), score=c.get("score", 0.0)) for c in raw]
    return RetrieveResponse(chunks=chunks)
