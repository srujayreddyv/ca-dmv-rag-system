"""Generate answers using an LLM with optional retrieved context (RAG).

Supports OpenAI and any OpenAI-compatible API (Ollama, Groq, OpenRouter, etc.)
via OPENAI_BASE_URL. See .env.example for free options.
"""

import os
from typing import Any, Iterator, Optional

from src.config.settings import LLM_MODEL

from .prompts import build_qa_prompt


def _get_client_and_model() -> tuple[Any, str]:
    """Return (OpenAI client, model name). Raises ValueError if not configured."""
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not base_url and not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Use OpenAI (paid), or Ollama/Groq/OpenRouter (free). See .env.example."
        )
    from openai import OpenAI
    model = os.getenv("LLM_MODEL") or LLM_MODEL
    client_kw = {"api_key": api_key or "ollama"} if base_url else {"api_key": api_key}
    if base_url:
        client_kw["base_url"] = base_url
    return OpenAI(**client_kw), model


def answer_stream(question: str, context_chunks: Optional[list[dict[str, Any]]] = None) -> Iterator[str]:
    """
    Stream answer tokens from the LLM. Yields text chunks (may be empty strings).
    Uses same config as answer().
    """
    if context_chunks:
        prompt = build_qa_prompt(question, context_chunks)
    else:
        prompt = f"Answer the following question:\n\n{question}"
    client, model = _get_client_and_model()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and getattr(delta, "content", None):
            yield delta.content


def answer(question: str, context_chunks: Optional[list[dict[str, Any]]] = None) -> str:
    """
    Produce an answer using the LLM. If context_chunks is provided, uses RAG.

    Uses OpenAI API by default. For Ollama, Groq, OpenRouter, etc., set
    OPENAI_BASE_URL (and OPENAI_API_KEY for non-Ollama). See .env.example.

    Args:
        question: User's question.
        context_chunks: Optional list of retrieved chunks. If None or empty, answers without RAG.

    Returns:
        Model-generated answer string.
    """
    client, model = _get_client_and_model()
    if context_chunks:
        prompt = build_qa_prompt(question, context_chunks)
    else:
        prompt = f"Answer the following question:\n\n{question}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def generate(prompt: str) -> str:
    """
    Send a custom prompt to the LLM and return the reply. Uses the same
    OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL as answer(). Useful for
    LLM-as-judge in evaluation.
    """
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not base_url and not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Use OpenAI (paid), or Ollama/Groq/OpenRouter (free). See .env.example."
        )
    from openai import OpenAI

    model = os.getenv("LLM_MODEL") or LLM_MODEL
    client_kw = {"api_key": api_key or "ollama"} if base_url else {"api_key": api_key}
    if base_url:
        client_kw["base_url"] = base_url
    return _generate(prompt, client_kw, model)


def _generate(prompt: str, client_kw: dict, model: str) -> str:
    from openai import OpenAI

    client = OpenAI(**client_kw)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""
