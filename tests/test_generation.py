"""Tests for generation: build_qa_prompt."""

from src.generation import build_qa_prompt


class TestBuildQaPrompt:
    def test_empty_context_returns_question_only(self):
        out = build_qa_prompt("What is the speed limit?", [])
        assert "Answer the following question" in out
        assert "What is the speed limit?" in out
        assert "Context:" not in out

    def test_empty_context_vs_none(self):
        # build_qa_prompt with [] falls through to "if not context_chunks" -> question only
        q = "X?"
        assert build_qa_prompt(q, []) == f"Answer the following question:\n\n{q}"

    def test_single_chunk_no_page(self):
        out = build_qa_prompt("Q?", [{"text": "The limit is 65 mph."}])
        assert "The limit is 65 mph." in out
        assert "California DMV" in out or "DMV" in out
        assert "I don't know" in out
        assert "Q?" in out
        assert "(Page" not in out

    def test_chunk_with_page(self):
        out = build_qa_prompt("Q?", [{"text": "Foo.", "page": 42}])
        assert "Foo." in out
        assert "(Page 42)" in out

    def test_multiple_chunks(self):
        out = build_qa_prompt(
            "Q?",
            [
                {"text": "One."},
                {"text": "Two.", "page": 2},
            ],
        )
        assert "One." in out and "Two." in out
        assert "(Page 2)" in out
        assert "---" in out

    def test_missing_text_uses_empty(self):
        out = build_qa_prompt("Q?", [{}])
        assert "Q?" in out
        # get("text","") gives ""; we'll have an empty segment; the join will still produce context
        assert "---" in out or "Question:" in out
