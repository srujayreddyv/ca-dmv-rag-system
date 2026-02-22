"""Vector store (FAISS) for embeddings and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorStore:
    """
    FAISS index (IndexFlatIP) with a JSON metadata file.
    Embeddings must be L2-normalized; scores are cosine similarity (higher = better).
    """

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """
        Add vectors and their metadata. Replaces any existing index.
        embeddings: (n, dim) float32, L2-normalized.
        """
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = list(metadata)

    def add_more(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """
        Append vectors and metadata to an existing index. Index must already exist (load or add first).
        embeddings: (n, dim) float32, L2-normalized, same dim as existing index.
        """
        if self.index is None:
            raise RuntimeError("No index loaded; call load() or add() first.")
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def save(self, path: str | Path) -> None:
        """Write FAISS index and handbook.meta.json next to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        meta_path = path.parent / (path.stem + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f)

    def load(self, path: str | Path) -> None:
        """Load FAISS index and metadata."""
        path = Path(path)
        self.index = faiss.read_index(str(path))
        meta_path = path.parent / (path.stem + ".meta.json")
        with open(meta_path) as f:
            self.metadata = json.load(f)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        query_embedding: (dim,) or (1, dim) float32, L2-normalized.
        Returns list of {**meta, "score": float}; score is cosine similarity.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = np.ascontiguousarray(query_embedding.astype(np.float32))
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        out = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx < 0 or idx >= len(self.metadata):
                continue
            row = dict(self.metadata[idx])
            row["score"] = float(scores[0][i])
            out.append(row)
        return out
