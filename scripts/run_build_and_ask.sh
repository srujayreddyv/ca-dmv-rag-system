#!/bin/bash
# Build the index and run ask.py. Run from project root:
#   ./scripts/run_build_and_ask.sh
#   ./scripts/run_build_and_ask.sh "What is the blood alcohol limit for DUI?"

set -e
cd "$(dirname "$0")/.."

echo "=== Building index (embedding chunks)... ==="
python scripts/build_index.py

echo ""
echo "=== Asking ==="
python scripts/ask.py "$@"
