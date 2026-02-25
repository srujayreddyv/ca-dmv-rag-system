"""Tests for retrieval: VectorStore. Optional slow tests for embed."""

import numpy as np
import pytest

from src.retrieval.vector_store import VectorStore
from src.retrieval.reranker import hybrid_rerank


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return (x / n).astype(np.float32)


class TestVectorStore:
    def test_add_and_search(self):
        dim = 4
        n = 3
        rng = np.random.default_rng(42)
        embs = _normalize(rng.random((n, dim)))
        meta = [{"text": f"chunk{i}", "page": i + 1} for i in range(n)]
        store = VectorStore()
        store.add(embs, meta)
        # query same as first vector -> should return it first
        q = embs[0:1]
        out = store.search(q, top_k=2)
        assert len(out) == 2
        assert out[0]["text"] == "chunk0"
        assert "score" in out[0]
        assert out[0]["score"] >= out[1]["score"]

    def test_search_query_1d(self):
        dim = 4
        embs = _normalize(np.random.randn(2, dim).astype(np.float32))
        store = VectorStore()
        store.add(embs, [{"text": "a"}, {"text": "b"}])
        q = embs[0]  # (dim,)
        out = store.search(q, top_k=2)
        assert len(out) == 2

    def test_search_top_k_greater_than_n(self):
        dim = 4
        embs = _normalize(np.random.randn(2, dim).astype(np.float32))
        store = VectorStore()
        store.add(embs, [{"text": "a"}, {"text": "b"}])
        out = store.search(embs[0:1], top_k=10)
        assert len(out) == 2

    def test_save_and_load(self, tmp_path):
        dim = 4
        embs = _normalize(np.random.randn(2, dim).astype(np.float32))
        meta = [{"text": "x", "page": 1}, {"text": "y", "page": 2}]
        store = VectorStore()
        store.add(embs, meta)
        path = tmp_path / "idx.faiss"
        store.save(path)
        assert path.exists()
        assert (tmp_path / "idx.meta.json").exists()
        store2 = VectorStore()
        store2.load(path)
        out = store2.search(embs[0:1], top_k=2)
        assert len(out) == 2
        assert out[0]["text"] in ("x", "y")

    def test_add_more(self):
        dim = 4
        rng = np.random.default_rng(99)
        embs1 = _normalize(rng.random((2, dim)))
        meta1 = [{"text": "a", "page": 1}, {"text": "b", "page": 2}]
        store = VectorStore()
        store.add(embs1, meta1)
        embs2 = _normalize(rng.random((2, dim)))
        meta2 = [{"text": "c", "page": 3}, {"text": "d", "page": 4}]
        store.add_more(embs2, meta2)
        assert store.index.ntotal == 4
        out = store.search(embs1[0:1], top_k=4)
        assert len(out) == 4
        texts = [o["text"] for o in out]
        assert "a" in texts and "b" in texts and "c" in texts and "d" in texts


class TestHybridRerank:
    def test_lexical_overlap_boosts_relevance(self):
        query = "blood alcohol limit dui"
        chunks = [
            {"text": "General parking information and permits.", "score": 0.95},
            {"text": "The blood alcohol concentration limit for DUI is 0.08%.", "score": 0.70},
        ]
        out = hybrid_rerank(query, chunks, top_k=2, alpha=0.6)
        assert len(out) == 2
        assert "blood alcohol" in out[0]["text"].lower()
        assert out[0]["hybrid_score"] >= out[1]["hybrid_score"]


@pytest.mark.slow
class TestEmbed:
    """Slow: loads sentence-transformers. Run with pytest -m 'not slow' to skip."""

    def test_embed_texts_shape(self):
        from src.retrieval import embed_texts

        out = embed_texts(["hello"])
        assert out.shape == (1, 384)  # all-MiniLM-L6-v2 dim
        assert out.dtype == np.float32

    def test_embed_query_shape(self):
        from src.retrieval import embed_query

        out = embed_query("hello")
        assert out.shape == (384,)
        assert out.dtype == np.float32
