"""Prompt templates for RAG-based question answering."""

from typing import Any


def build_qa_prompt(question: str, context_chunks: list[dict[str, Any]]) -> str:
    """
    Build a prompt with instructions, retrieved context, and the question.

    Args:
        question: User's question.
        context_chunks: List of dicts with "text" and optionally "page".

    Returns:
        Formatted prompt string for the LLM.
    """
    if not context_chunks:
        return f"Answer the following question:\n\n{question}"

    parts = []
    for c in context_chunks:
        t = c.get("text", "")
        page = c.get("page")
        if page is not None:
            t = f"{t}\n(Page {page})"
        parts.append(t)

    context = "\n\n---\n\n".join(parts)
    return f"""You are a helpful assistant. Answer the question using only the following context from the California DMV Driver Handbook. If the context does not contain the answer, say "I don't know."

Context:

{context}

---

Question: {question}"""
