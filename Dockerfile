# CA DMV RAG API & Streamlit — single image, run as api or streamlit via compose
FROM python:3.11-slim

WORKDIR /app

# Install dependencies (sentence-transformers and faiss need build deps for some platforms)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build chunks and FAISS index at image build time (so Render/deploy has the index).
RUN python scripts/build_chunks.py && python scripts/build_index.py

# Default: run API. Override in docker-compose for streamlit.
# Use PORT from env so Render can inject it (e.g. 10000).
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
