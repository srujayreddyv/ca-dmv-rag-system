#!/usr/bin/env python3
"""
Append new chunks to an existing FAISS index (incremental index update).

Run from project root (after handbook.index exists):
    python scripts/append_index.py data/processed/append_chunks.jsonl

Creates append_chunks.jsonl with one JSON object per line (same format as chunks.jsonl:
{"text": "...", "page": 1, "source": "optional"}).
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import INDEX_PATH
from src.retrieval import embed_texts
from src.retrieval.vector_store import VectorStore


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/append_index.py <path-to-new-chunks.jsonl>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    if not INDEX_PATH.exists():
        print(f"Index not found: {INDEX_PATH}. Run: python scripts/build_index.py")
        sys.exit(1)

    chunks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    if not chunks:
        print("No chunks in file. Nothing to append.")
        sys.exit(0)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} new chunks...")
    embeddings = embed_texts(texts)

    store = VectorStore()
    store.load(INDEX_PATH)
    store.add_more(embeddings, chunks)
    store.save(INDEX_PATH)
    meta_path = INDEX_PATH.parent / (INDEX_PATH.stem + ".meta.json")
    print(f"Appended {len(chunks)} chunks. Saved to {INDEX_PATH} and {meta_path} (total vectors: {store.index.ntotal})")


if __name__ == "__main__":
    main()
