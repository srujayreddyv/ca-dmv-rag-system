"""Tests for ingestion: clean_text, chunk_text, load_pdf."""

import pytest
from pypdf import PdfWriter

from src.ingestion import clean_text, chunk_text, load_pdf


class TestCleanText:
    def test_empty(self):
        assert clean_text("") == ""

    def test_strip(self):
        assert clean_text("  foo  ") == "foo"

    def test_collapse_spaces(self):
        assert clean_text("a   b\t\tc") == "a b c"

    def test_collapse_newlines(self):
        assert clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_mixed(self):
        assert clean_text("  hi \t \n\n\n world  ") == "hi \n\n world"


class TestChunkText:
    def test_empty(self):
        assert chunk_text("", chunk_size=100, overlap=10) == []

    def test_shorter_than_chunk(self):
        out = chunk_text("abc", chunk_size=10, overlap=2)
        assert len(out) == 1
        assert out[0]["text"] == "abc"
        assert out[0]["start"] == 0
        assert out[0]["end"] == 3

    def test_exact_two_chunks(self):
        text = "a" * 100
        out = chunk_text(text, chunk_size=50, overlap=0)
        assert len(out) == 2
        assert out[0]["text"] == "a" * 50 and out[0]["end"] == 50
        assert out[1]["text"] == "a" * 50 and out[1]["start"] == 50

    def test_overlap(self):
        text = "abcdefghij"  # 10 chars
        out = chunk_text(text, chunk_size=5, overlap=2)
        assert len(out) >= 2
        assert out[0]["text"] == "abcde"
        assert out[1]["start"] == 3  # 5 - 2
        assert out[1]["text"] == "defgh"
        # third: start 6, "ghi" + 2 from overlap -> "ghij" if we have 4? chunk 6:6+5=11, so "ghij" (len 4)
        # start += 5-2=3, so 0,3,6,9. at 9: end=min(14,10)=10, "j". So 4 chunks.
        assert out[0]["end"] - out[0]["start"] <= 5
        assert out[1]["end"] - out[1]["start"] <= 5

    def test_defaults(self):
        text = "x" * 600
        out = chunk_text(text)
        assert len(out) >= 1
        assert all("text" in c and "start" in c and "end" in c for c in out)


class TestLoadPdf:
    def test_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            load_pdf("/nonexistent/path.pdf")

    def test_minimal_pdf(self, tmp_path):
        path = tmp_path / "sample.pdf"
        w = PdfWriter()
        w.add_blank_page(200, 200)
        with open(path, "wb") as f:
            w.write(f)
        out = load_pdf(path)
        assert len(out) == 1
        assert out[0]["page"] == 1
        assert "text" in out[0]
        assert isinstance(out[0]["text"], str)

    def test_handbook_if_present(self):
        from src.config.settings import PDF_PATH

        if not PDF_PATH.exists():
            pytest.skip("handbook PDF not found")
        out = load_pdf(PDF_PATH)
        assert len(out) >= 1
        for p in out:
            assert "page" in p and "text" in p
            assert isinstance(p["page"], int)
            assert isinstance(p["text"], str)
