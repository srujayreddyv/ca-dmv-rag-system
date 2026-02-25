"""Rerank retrieved chunks with a cross-encoder (optional)."""

from __future__ import annotations

import re
from typing import Any

_reranker_model = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(RERANKER_MODEL)
    return _reranker_model


def rerank(query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """
    Rerank chunks by cross-encoder score (query, chunk text). Returns top_k chunks with updated "score".
    """
    if not chunks or top_k <= 0:
        return chunks[:top_k]
    model = _get_reranker()
    pairs = [(query, c.get("text", "")[:512]) for c in chunks]  # truncate long chunks for model
    scores = model.predict(pairs)
    indexed = [(float(scores[i]), chunks[i]) for i in range(len(chunks))]
    indexed.sort(key=lambda x: -x[0])
    out = []
    for score, c in indexed[:top_k]:
        row = dict(c)
        row["score"] = score
        out.append(row)
    return out


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def hybrid_rerank(query: str, chunks: list[dict[str, Any]], top_k: int = 5, alpha: float = 0.75) -> list[dict[str, Any]]:
    """
    Lightweight reranking without extra models.
    Combines normalized semantic score with lexical overlap score.
    Keeps original semantic `score` unchanged and stores blend in `hybrid_score`.
    """
    if not chunks or top_k <= 0:
        return chunks[:top_k]

    alpha = max(0.0, min(1.0, float(alpha)))
    q_tokens = set(_tokenize(query))
    sem_scores = [float(c.get("score", 0.0)) for c in chunks]
    s_min, s_max = min(sem_scores), max(sem_scores)
    s_rng = (s_max - s_min) or 1.0

    scored = []
    for c in chunks:
        sem = (float(c.get("score", 0.0)) - s_min) / s_rng
        t = set(_tokenize(c.get("text", "")))
        lex = (len(q_tokens & t) / len(q_tokens)) if q_tokens else 0.0
        # Lexical bonus helps exact keyword matches beat semantically-near but off-topic chunks.
        final = alpha * sem + (1.0 - alpha) * lex + 0.30 * lex
        row = dict(c)
        row["hybrid_score"] = float(final)
        scored.append(row)

    scored.sort(key=lambda x: -x.get("hybrid_score", 0.0))
    return scored[:top_k]
