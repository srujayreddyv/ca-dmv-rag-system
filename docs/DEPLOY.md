# Deploying the CA DMV RAG API

You can run the app locally (see README) or deploy the **API** to a cloud host. The Streamlit UI can stay local or be deployed separately (e.g. Streamlit Cloud).

---

## Deploy the API to Render

Render runs the API as a **web service** from your Git repo. The build step creates the handbook index; the start step runs the FastAPI app.

### 1. Prerequisites

- **Git repo** with this project pushed to GitHub, GitLab, or Bitbucket.
- **PDF in the repo** so the build can create the index: ensure `data/ca-drivers-handbook.pdf` is committed. If it’s not in the repo, the build will fail unless you use a [persistent disk](https://render.com/docs/disks) and add the PDF there in a custom build step.
- **LLM API key** (OpenAI, Groq, OpenRouter, etc.) to set in Render’s environment.

### 2. One-time setup on Render

1. Go to [dashboard.render.com](https://dashboard.render.com) and sign in.
2. **New** → **Blueprint** (or **Web Service** if you prefer to configure by hand).
3. Connect your Git repo and choose this project.
4. If you use the **Blueprint** (`render.yaml` in the repo):
   - Render will create a service named `ca-dmv-rag-api` with the build/start commands from `render.yaml`.
5. If you create a **Web Service** manually:
   - **Runtime:** Python 3.11.
   - **Build command:**  
     `pip install -r requirements.txt && python scripts/build_chunks.py && python scripts/build_index.py`
   - **Start command:**  
     `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

### 3. Environment variables (required for the LLM)

In the Render service → **Environment** tab, add at least:

- **OPENAI_API_KEY** — Your API key (OpenAI, Groq, OpenRouter, etc.).
- **LLM_MODEL** — e.g. `gpt-4o-mini`, `llama-3.1-70b-versatile`, `google/gemma-2-9b-it:free`.

If you use a non-OpenAI endpoint:

- **OPENAI_BASE_URL** — e.g. `https://api.groq.com/openai/v1` or `https://openrouter.ai/api/v1`.

See `.env.example` for all options.

### 4. Optional: rate limiting (for a public API)

To limit abuse, set:

- **RATE_LIMIT_REQUESTS** — e.g. `60`.
- **RATE_LIMIT_WINDOW_SECONDS** — e.g. `60`.

So each IP gets 60 requests per 60 seconds. Leave unset or set to `0` to disable.

### 5. Deploy

- **Blueprint:** Push to your repo; Render will deploy from `render.yaml`.
- **Manual:** Click **Deploy** (or push to the connected branch).

After deploy, the API will be at `https://<your-service>.onrender.com`. Use:

- `GET /health` — health check  
- `POST /ask` — RAG question  
- `POST /ask/stream` — streaming answer  
- `GET /docs` — Swagger UI  

### 6. Using the Streamlit app with the deployed API

Point the app at the deployed API:

- **Local Streamlit:** Set `API_URL=https://<your-service>.onrender.com` in your `.env`, then run `streamlit run streamlit_app.py`.
- **Streamlit Cloud:** In the app’s secrets, set `API_URL` to `https://<your-service>.onrender.com`.

---

## Other hosts (Fly.io, Railway, etc.)

The app is a standard FastAPI app (Python 3.10+; 3.11 recommended):

- **Build:** Install deps and build the index (e.g. `pip install -r requirements.txt && python scripts/build_chunks.py && python scripts/build_index.py`).
- **Run:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` (use the host’s `PORT` env var).
- **Env:** Same as above (OPENAI_API_KEY, LLM_MODEL, optional OPENAI_BASE_URL, RATE_LIMIT_*).

Ensure `data/ca-drivers-handbook.pdf` is available at build time (in the image or via a volume) so the index can be built.

---

## CI (GitHub Actions)

The repo includes `.github/workflows/ci.yml`:

- **test:** Runs on every push/PR; installs deps and runs `pytest tests/ -m "not slow"`.
- **eval:** Runs on push to `main`/`master` only; builds chunks and index from the PDF, then runs `run_eval.py --no-save`. The eval job uses **Secrets** for the LLM:
  - **OPENAI_API_KEY** (required for eval) — Set in repo **Settings → Secrets and variables → Actions**.
  - **OPENAI_BASE_URL**, **LLM_MODEL** (optional) — For non-OpenAI endpoints.

If the PDF or `data/eval/questions.json` is missing, or the LLM secret is not set, the eval job is configured to `continue-on-error` so CI does not fail.
