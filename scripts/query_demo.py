#!/usr/bin/env python3
"""
Demo: run a single query through the retriever.

Run from project root (after scripts/build_index.py):
    python scripts/query_demo.py
    python scripts/query_demo.py "What is the speed limit on a highway?"
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import INDEX_PATH, TOP_K
from src.retrieval import Retriever


def main() -> None:
    if not INDEX_PATH.exists():
        print(f"Index not found: {INDEX_PATH}. Run: python scripts/build_index.py")
        sys.exit(1)

    query = sys.argv[1] if len(sys.argv) > 1 else "What is the blood alcohol limit for DUI?"
    r = Retriever(INDEX_PATH)
    chunks = r.retrieve(query, top_k=TOP_K)

    print(f"Query: {query}\n")
    print(f"Top {len(chunks)} chunks:\n")
    for i, c in enumerate(chunks, 1):
        print(f"--- {i} (page {c.get('page')}, score={c.get('score', 0):.4f}) ---")
        print(c["text"][:400] + "..." if len(c["text"]) > 400 else c["text"])
        print()


if __name__ == "__main__":
    main()
