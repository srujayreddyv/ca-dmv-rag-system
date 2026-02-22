#!/usr/bin/env python3
"""
Ask a question: retrieve top-k chunks and generate an answer with the LLM.

Run from project root (after build_index, with OPENAI_API_KEY set):
    python scripts/ask.py
    python scripts/ask.py "What is the blood alcohol limit for DUI?"
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import AuthenticationError, NotFoundError, RateLimitError

from src.config.settings import INDEX_PATH, TOP_K
from src.generation import answer
from src.retrieval import Retriever


def main() -> None:
    if not INDEX_PATH.exists():
        print(f"Index not found: {INDEX_PATH}. Run: python scripts/build_index.py")
        sys.exit(1)

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the blood alcohol limit for DUI?"
    )

    retriever = Retriever(INDEX_PATH)
    chunks = retriever.retrieve(question, top_k=TOP_K)

    print(f"Question: {question}\n")
    print("Answer:")
    try:
        out = answer(question, context_chunks=chunks)
        print(out)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except AuthenticationError:
        print("Invalid API key. Use a valid key or Ollama/Groq/OpenRouter. See .env.example.")
        sys.exit(1)
    except NotFoundError:
        print("Model not found. For Ollama: run  ollama list  to see models; if empty, run  ollama pull llama3.2 . Then set LLM_MODEL in .env to that exact name.")
        sys.exit(1)
    except RateLimitError:
        print("Quota exceeded (429). Use a paid OpenAI plan or Ollama/Groq/OpenRouter. See .env.example.")
        sys.exit(1)


if __name__ == "__main__":
    main()
