#!/bin/bash
# Remove Python cache, pytest cache, and macOS cruft (not .venv or .env).
# Run from project root: ./scripts/clean.sh

set -e
cd "$(dirname "$0")/.."

echo "Removing __pycache__, .pytest_cache, .ipynb_checkpoints, .DS_Store..."
find . -depth -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -depth -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
find . -depth -type d -name .ipynb_checkpoints -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
find . -name .DS_Store -delete 2>/dev/null || true
echo "Optional: data/embeddings and data/eval/results are in .gitignore; delete to regenerate from scratch."
echo "Done."
