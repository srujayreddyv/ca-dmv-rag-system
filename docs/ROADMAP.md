# CA DMV RAG — Roadmap

What’s done and what to add next. Use this to plan and prioritize.

---

## Done

- Full RAG pipeline: ingestion → retrieval → generation
- API (`/ask`, `/retrieve`, `/health`, `/docs`) + Streamlit UI
- **Sources in responses** — `/ask` returns `sources` (page, score, snippet); Streamlit shows “Sources (from handbook vector DB)”
- Evaluation: exact match, ref-in-pred, optional LLM judge
- Tests, notebooks, README, setup docs
- **Quick wins (Phase 1):** Low-confidence handling (`SCORE_THRESHOLD`, `confidence` in API); persist eval results (`data/eval/results/`, `--no-save`); input limits (max 1000 chars, 422).
- **Phase 2 — Polish:** Conversation history in Streamlit (session list + “Clear history”); Docker (Dockerfile + docker-compose for API + Streamlit, volume `./data`); structured logging for `/ask` (question_len, confidence, latency_seconds, errors).
- **Phase 3 — Scale & share:** Streaming answers; deploy; rate limiting.
- **Reranker:** Optional cross-encoder rerank (`USE_RERANKER`, `RERANK_RETRIEVE_K`); used in API and eval.
- **Extra eval metrics:** `embedding_similarity` (ref vs pred); persisted in results JSON.
- **Eval in CI:** `.github/workflows/ci.yml` runs `pytest -m "not slow"` on push/PR.
- **include_sources / source_filter:** `POST /ask` and `/ask/stream` accept `include_sources` (bool) and `source_filter` (optional); `POST /retrieve` accepts `source_filter`.
- **Multiple PDFs:** `PDF_PATHS` (env: comma-separated paths under data/); each chunk has `source` (filename stem).
- **Semantic chunking:** `USE_SEMANTIC_CHUNKING=1` uses paragraph-aware chunking (`chunk_by_paragraphs`).

---

## Next: by category

### 1. Quality & safety

| Item | What | Effort | Impact |
|------|------|--------|--------|
| ~~**Low-confidence handling**~~ | ✅ Done. | — | —
| ~~**Input limits**~~ | ✅ Done (max 1000 chars, 422). | — | — |
| ~~**Reranker**~~ | ✅ Done. USE_RERANKER, RERANK_RETRIEVE_K; cross-encoder rerank. | — | — |

### 2. UX & transparency

| Item | What | Effort | Impact |
|------|------|--------|--------|
| ~~**Conversation history**~~ | ✅ Done. Session list + Clear history in sidebar. | — | — |
| ~~**Streaming answers**~~ | ✅ Done. POST /ask/stream (SSE); Streamlit “Stream answer” checkbox. | — | — |
| ~~**Optional include_sources**~~ | ✅ Done. include_sources + source_filter on /ask, /ask/stream, /retrieve. | — | — |

### 3. Evaluation & iteration

| Item | What | Effort | Impact |
|------|------|--------|--------|
| ~~**Persist eval results**~~ | ✅ Done. `run_eval.py` writes timestamped + `latest.json`; `--no-save` to skip. | — | — |
| ~~**Extra metrics**~~ | ✅ Done. embedding_similarity in run_eval + persisted. | — | — |
| ~~**Eval in CI**~~ | ✅ Done. .github/workflows/ci.yml runs pytest (not slow). | — | — |

### 4. Deployment & ops

| Item | What | Effort | Impact |
|------|------|--------|--------|
| ~~**Docker**~~ | ✅ Done. Dockerfile + docker-compose (API + Streamlit, volume `./data`). | — | — |
| ~~**Deploy**~~ | ✅ Done. render.yaml + docs/DEPLOY.md (Render and other hosts). | — | — |
| ~~**Logging**~~ | ✅ Done. `/ask` logs question_len, confidence, latency_seconds, errors. | — | — |
| ~~**Rate limiting**~~ | ✅ Done. Per-IP for /ask and /ask/stream; RATE_LIMIT_* (0 = off). | — | — |

### 5. Data & ingestion (optional)

| Item | What | Effort | Impact |
|------|------|--------|--------|
| ~~**Multiple PDFs**~~ | ✅ Done. PDF_PATHS env; source on chunks; source_filter in API. | — | — |
| ~~**Semantic chunking**~~ | ✅ Done. USE_SEMANTIC_CHUNKING=1 → chunk_by_paragraphs. | — | — |
| **Incremental index** | Add new chunks without full rebuild (FAISS supports add). | Medium | Low — only if you often add content |

---

## Suggested order

**Phase 1 — Quick wins** ✅ Done  
1. ~~Low-confidence handling~~  
2. ~~Persist eval results~~  
3. ~~Input limits (max question length)~~

**Phase 2 — Polish** ✅ Done  
4. ~~Conversation history in Streamlit~~  
5. ~~Docker (API + optional Streamlit)~~  
6. ~~Logging~~

**Phase 3 — Scale & share** ✅ Done  
7. ~~Streaming answers~~  
8. ~~Deploy (e.g. Render)~~  
9. ~~Rate limiting (if public)~~

**All roadmap items implemented.**

**Optional enhancements (done):** Full eval in CI (eval job on push to main; needs `OPENAI_API_KEY` secret); incremental index (`scripts/append_index.py`); BLEU + ROUGE-L in eval and persisted; Streamlit **Include sources** and **Source filter** in sidebar.

---

## How to use this doc

- Check off items when done; add your own and reorder by priority.
- Link to issues or PRs next to each item if you use GitHub.

## What to do next (suggested)

1. **Run and test** — Use the app locally; try `USE_RERANKER=1`, `USE_SEMANTIC_CHUNKING=1`, or multiple PDFs.
2. **Deploy when ready** — Follow [DEPLOY.md](DEPLOY.md).
3. **CI** — Push to GitHub; workflow runs tests on push/PR.
