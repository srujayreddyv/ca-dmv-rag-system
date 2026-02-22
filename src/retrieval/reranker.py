"""Rerank retrieved chunks with a cross-encoder (optional)."""

from __future__ import annotations

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
