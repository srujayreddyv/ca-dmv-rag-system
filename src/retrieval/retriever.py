"""Retriever: embed query and return top-k chunks from the vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .embed import embed_query
from .vector_store import VectorStore


class Retriever:
    def __init__(self, index_path: str | Path) -> None:
        self.store = VectorStore()
        self.store.load(index_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Return the top-k most relevant chunks for the query.
        Each dict has "text", "page", "score", and optionally "start", "end".
        """
        q = embed_query(query)
        return self.store.search(q, top_k=top_k)
