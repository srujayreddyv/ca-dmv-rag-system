"""Integration-style API tests using a real FAISS index and retriever path."""

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.retrieval.vector_store import VectorStore


def _build_fixture_index(index_path):
    store = VectorStore()
    embeddings = np.ascontiguousarray(np.array([[1.0, 0.0]], dtype=np.float32))
    metadata = [{"text": "Fixture chunk about speed limits.", "page": 7, "source": "fixture"}]
    store.add(embeddings, metadata)
    store.save(index_path)


def test_ask_real_retriever_low_confidence_path(tmp_path):
    index_path = tmp_path / "handbook.index"
    chunks_path = tmp_path / "chunks.jsonl"
    _build_fixture_index(index_path)

    with (
        patch("src.api.main.INDEX_PATH", index_path),
        patch("src.api.main.CHUNKS_JSONL", chunks_path),
        patch("src.api.main.SCORE_THRESHOLD", 1.1),
        patch("src.api.main._retriever", None),
        patch("src.retrieval.retriever.embed_query", return_value=np.array([1.0, 0.0], dtype=np.float32)),
    ):
        with TestClient(app) as client:
            r = client.post("/ask", json={"question": "What is the speed limit?"})
            assert r.status_code == 200
            body = r.json()
            assert body["confidence"] == "low"
            assert "don't have enough" in body["answer"].lower() or "not enough" in body["answer"].lower()
            assert body["sources"]

            m = client.get("/metrics")
            assert m.status_code == 200
            metrics = m.json()
            assert metrics["requests_total"] >= 2
            assert metrics["tokens_estimated"]["prompt_total"] > 0


def test_ask_stream_real_retriever_low_confidence_path(tmp_path):
    index_path = tmp_path / "handbook.index"
    chunks_path = tmp_path / "chunks.jsonl"
    _build_fixture_index(index_path)

    with (
        patch("src.api.main.INDEX_PATH", index_path),
        patch("src.api.main.CHUNKS_JSONL", chunks_path),
        patch("src.api.main.SCORE_THRESHOLD", 1.1),
        patch("src.api.main._retriever", None),
        patch("src.retrieval.retriever.embed_query", return_value=np.array([1.0, 0.0], dtype=np.float32)),
    ):
        with TestClient(app) as client:
            with client.stream("POST", "/ask/stream", json={"question": "What is the speed limit?"}) as r:
                assert r.status_code == 200
                lines = [ln.decode() if isinstance(ln, bytes) else ln for ln in r.iter_lines() if ln]
            events = [ln for ln in lines if ln.startswith("data: ")]
            assert events
            assert any('"done": true' in e.lower() for e in events)

