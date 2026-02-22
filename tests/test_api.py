"""Tests for the FastAPI app: GET /, GET /health, POST /ask, POST /retrieve.

Uses mocks for _build_retriever and answer to avoid loading sentence-transformers or the LLM.
Run: pytest tests/test_api.py -v -m "not slow"
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class FakeRetriever:
    def retrieve(self, q, top_k=5):
        return [
            {"text": "chunk1", "page": 1, "score": 0.9},
            {"text": "chunk2", "page": 2, "score": 0.8},
        ]


@patch("src.api.main.CHUNKS_JSONL")
@patch("src.api.main.INDEX_PATH")
def test_root_returns_links(mock_index_path, mock_chunks_jsonl):
    mock_index_path.exists.return_value = True
    mock_chunks_jsonl.exists.return_value = False
    r = client.get("/")
    assert r.status_code == 200
    j = r.json()
    assert j["name"] == "CA DMV RAG API"
    assert j["docs"] == "/docs"
    assert j["redoc"] == "/redoc"
    assert j["openapi"] == "/openapi.json"
    assert j["status"] == "ok"
    assert "health" in j["endpoints"] and "ask" in j["endpoints"] and "retrieve" in j["endpoints"]


@patch("src.api.main.CHUNKS_JSONL")
@patch("src.api.main.INDEX_PATH")
def test_root_status_degraded_when_neither_exist(mock_index_path, mock_chunks_jsonl):
    mock_index_path.exists.return_value = False
    mock_chunks_jsonl.exists.return_value = False
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


@patch("src.api.main.CHUNKS_JSONL")
@patch("src.api.main.INDEX_PATH")
def test_health_ok_when_index_or_chunks_exist(mock_index_path, mock_chunks_jsonl):
    mock_index_path.exists.return_value = True
    mock_chunks_jsonl.exists.return_value = False
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@patch("src.api.main.CHUNKS_JSONL")
@patch("src.api.main.INDEX_PATH")
def test_health_503_when_neither_exist(mock_index_path, mock_chunks_jsonl):
    mock_index_path.exists.return_value = False
    mock_chunks_jsonl.exists.return_value = False
    r = client.get("/health")
    assert r.status_code == 503
    assert "detail" in r.json()


@patch("src.api.main._build_retriever")
def test_retrieve_ok(mock_build):
    mock_build.return_value = FakeRetriever()
    r = client.post("/retrieve", json={"question": "What is the speed limit?", "top_k": 3})
    assert r.status_code == 200
    j = r.json()
    assert "chunks" in j
    assert len(j["chunks"]) == 2
    assert j["chunks"][0]["text"] == "chunk1"
    assert j["chunks"][0]["page"] == 1
    assert j["chunks"][0]["score"] == 0.9


@patch("src.api.main._build_retriever")
def test_retrieve_default_top_k(mock_build):
    mock_build.return_value = FakeRetriever()
    r = client.post("/retrieve", json={"question": "x"})
    assert r.status_code == 200
    assert len(r.json()["chunks"]) == 2


@patch("src.api.main._build_retriever")
def test_retrieve_503_when_build_fails(mock_build):
    mock_build.side_effect = FileNotFoundError("Need chunks or index")
    r = client.post("/retrieve", json={"question": "x"})
    assert r.status_code == 503
    assert "chunks" in r.json()["detail"].lower() or "need" in r.json()["detail"].lower()


def test_retrieve_validates_top_k():
    r = client.post("/retrieve", json={"question": "x", "top_k": 0})
    assert r.status_code == 422
    r = client.post("/retrieve", json={"question": "x", "top_k": 21})
    assert r.status_code == 422


@patch("src.api.main.answer")
@patch("src.api.main._build_retriever")
def test_ask_ok(mock_build, mock_answer):
    mock_build.return_value = FakeRetriever()
    mock_answer.return_value = "A test answer."
    r = client.post("/ask", json={"question": "What is the speed limit?"})
    assert r.status_code == 200
    j = r.json()
    assert j["answer"] == "A test answer."
    assert "sources" in j
    assert j.get("confidence") == "high"


@patch("src.api.main.answer")
@patch("src.api.main._build_retriever")
def test_ask_empty_question_uses_default(mock_build, mock_answer):
    mock_build.return_value = FakeRetriever()
    mock_answer.return_value = "Answer."
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 200
    assert r.json()["answer"] == "Answer."


@patch("src.api.main._build_retriever")
def test_ask_503_when_build_fails(mock_build):
    mock_build.side_effect = FileNotFoundError("Need chunks or index")
    r = client.post("/ask", json={"question": "x"})
    assert r.status_code == 503


@patch("src.api.main.answer")
@patch("src.api.main._build_retriever")
def test_ask_503_when_llm_not_configured(mock_build, mock_answer):
    mock_build.return_value = FakeRetriever()
    mock_answer.side_effect = ValueError("LLM not configured")
    r = client.post("/ask", json={"question": "x"})
    assert r.status_code == 503
    assert "LLM" in r.json()["detail"]


class LowScoreRetriever:
    def retrieve(self, q, top_k=5):
        return [
            {"text": "chunk1", "page": 1, "score": 0.1},
            {"text": "chunk2", "page": 2, "score": 0.2},
        ]


@patch("src.api.main.answer")
@patch("src.api.main._build_retriever")
def test_ask_low_confidence_skips_llm(mock_build, mock_answer):
    mock_build.return_value = LowScoreRetriever()
    r = client.post("/ask", json={"question": "What is the speed limit?"})
    assert r.status_code == 200
    j = r.json()
    assert j.get("confidence") == "low"
    assert "don't have enough" in j["answer"].lower() or "not enough" in j["answer"].lower()
    mock_answer.assert_not_called()


def test_ask_422_when_question_too_long():
    r = client.post("/ask", json={"question": "x" * 1001})
    assert r.status_code == 422


@patch("src.api.main.answer")
@patch("src.api.main._build_retriever")
def test_ask_include_sources_false(mock_build, mock_answer):
    mock_build.return_value = FakeRetriever()
    mock_answer.return_value = "Answer."
    r = client.post("/ask", json={"question": "x", "include_sources": False})
    assert r.status_code == 200
    assert r.json()["answer"] == "Answer."
    assert r.json()["sources"] == []


class RetrieverWithSource:
    def retrieve(self, q, top_k=5):
        return [
            {"text": "chunk1", "page": 1, "score": 0.9, "source": "handbook"},
            {"text": "chunk2", "page": 2, "score": 0.8, "source": "other"},
        ]


@patch("src.api.main.answer")
@patch("src.api.main._build_retriever")
def test_ask_source_filter(mock_build, mock_answer):
    mock_build.return_value = RetrieverWithSource()
    mock_answer.return_value = "Answer."
    r = client.post("/ask", json={"question": "x", "source_filter": "handbook"})
    assert r.status_code == 200
    _, kwargs = mock_answer.call_args
    context = kwargs.get("context_chunks", [])
    assert len(context) == 1
    assert context[0].get("source") == "handbook"


@patch("src.api.main._build_retriever")
def test_retrieve_source_filter(mock_build):
    mock_build.return_value = RetrieverWithSource()
    r = client.post("/retrieve", json={"question": "x", "source_filter": "handbook"})
    assert r.status_code == 200
    assert len(r.json()["chunks"]) == 1
    assert r.json()["chunks"][0]["text"] == "chunk1"
