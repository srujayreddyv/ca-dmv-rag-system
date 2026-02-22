#!/bin/sh
set -e

if [ ! -f /app/data/embeddings/handbook.index ]; then
  python scripts/build_chunks.py
  python scripts/build_index.py
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-10000}"
