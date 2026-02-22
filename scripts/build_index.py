#!/usr/bin/env python3
"""
Build the vector index from chunks.jsonl: embed all chunks and save to handbook.index.

Run from project root:
    python scripts/build_index.py

Requires: data/processed/chunks.jsonl (from scripts/build_chunks.py)
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import CHUNKS_JSONL, INDEX_PATH
from src.retrieval import embed_texts
from src.retrieval.vector_store import VectorStore


def main() -> None:
    if not CHUNKS_JSONL.exists():
        print(f"Missing {CHUNKS_JSONL}. Run: python scripts/build_chunks.py")
        sys.exit(1)

    chunks = []
    with open(CHUNKS_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    store = VectorStore()
    store.add(embeddings, chunks)
    store.save(INDEX_PATH)
    meta = INDEX_PATH.parent / (INDEX_PATH.stem + ".meta.json")
    print(f"Saved to {INDEX_PATH} and {meta}")
