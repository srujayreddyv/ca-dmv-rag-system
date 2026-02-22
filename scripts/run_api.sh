#!/bin/bash
# Run the FastAPI app. Start from project root:
#   ./scripts/run_api.sh
#
# Then run Streamlit: streamlit run streamlit_app.py

set -e
cd "$(dirname "$0")/.."

echo "Starting API at http://localhost:8000"
echo "  /docs     - Swagger UI"
echo "  /redoc    - ReDoc"
echo "  /health   - health check"
echo "  /ask      - POST {\"question\": \"...\"}"
echo "  /retrieve - POST {\"question\": \"...\", \"top_k\": 5}"
echo ""
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
