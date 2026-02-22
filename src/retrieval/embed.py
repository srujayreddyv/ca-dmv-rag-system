"""Generate embeddings for text (chunks and queries)."""

from __future__ import annotations

import numpy as np

from src.config.settings import EMBEDDING_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts; returns (n, dim) float32, L2-normalized.
    """
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query; returns (dim,) float32, L2-normalized.
    """
    arr = embed_texts([query])
    return arr[0]
