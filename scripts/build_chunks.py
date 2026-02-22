#!/usr/bin/env python3
"""
Build chunks.jsonl from the handbook PDF.

Run from project root:
    python scripts/build_chunks.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_JSONL,
    PDF_PATHS,
    USE_SEMANTIC_CHUNKING,
)
from src.ingestion import load_pdf, clean_text, chunk_text, chunk_by_paragraphs


def main() -> None:
    chunk_fn = chunk_by_paragraphs if USE_SEMANTIC_CHUNKING else chunk_text
    chunk_kw = (
        {"chunk_size": CHUNK_SIZE, "overlap_paragraphs": 1}
        if USE_SEMANTIC_CHUNKING
        else {"chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}
    )
    chunks = []
    for pdf_path in PDF_PATHS:
        if not pdf_path.exists():
            print(f"Warning: PDF not found {pdf_path}, skipping.")
            continue
        pages = load_pdf(pdf_path)
        source = pdf_path.stem
        for p in pages:
            cleaned = clean_text(p["text"])
            for ch in chunk_fn(cleaned, **chunk_kw):
                row = {**ch, "page": p["page"], "source": source}
                chunks.append(row)

    CHUNKS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSONL, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print(f"Wrote {len(chunks)} chunks to {CHUNKS_JSONL}")


if __name__ == "__main__":
    main()
