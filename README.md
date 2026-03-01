<!-- Created by Codex - Section 2 -->

# PubMed Literature Assistant

FastAPI + Angular application for biomedical literature Q&A grounded in PubMed abstracts.

## Local Run

### Backend

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from the example file and set `NVIDIA_API_KEY` if you want answer generation enabled.
4. Start the API:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

1. Install frontend dependencies:

```bash
cd frontend
npm ci
```

2. Start the Angular development server:

```bash
npm start
```

The Angular dev server runs on `http://localhost:4200` and proxies `/api` to `http://localhost:8000`.

## Tests

Backend:

```bash
pytest -q
```

Frontend:

```bash
cd frontend
npm test -- --watch=false --browsers=ChromeHeadless
```

Network smoke tests are opt-in:

```bash
$env:RUN_NETWORK_TESTS="true"
pytest -q tests/test_retrieval_smoke.py
```

## Core Features

- PubMed-grounded retrieval and answer generation with NVIDIA-hosted chat models.
- FastAPI invoke + SSE streaming endpoints for chat, session history, branching, paper fetch, and config introspection.
- Angular 17 single-page app with light/dark theme toggle, persisted client controls, and Claude-style branch navigation.
- Similar-query answer cache, PubMed query caching, and Chroma-backed abstract storage.
- Optional evaluation and metrics pipelines, plus branch/session persistence in `data/sessions.json`.

## Key Configuration

- `NVIDIA_API_KEY`: required for LLM answer generation.
- `NVIDIA_MODEL`: generation model name.
- `DATA_DIR`: root directory for Chroma persistence and local data.
- `API_PORT`: backend listen port for local/dev deployments.
- `FRONTEND_ORIGIN`: allowed browser origin for CORS.
- `PUBMED_CACHE_TTL_SECONDS`
- `PUBMED_NEGATIVE_CACHE_TTL_SECONDS`
- `MAX_CONTEXT_ABSTRACTS`
- `MAX_ABSTRACTS` (deprecated alias for `MAX_CONTEXT_ABSTRACTS`)
- `MAX_CONTEXT_TOKENS`
- `CONTEXT_TRIM_STRATEGY=truncate|compress`
- `HYBRID_RETRIEVAL=true|false`
- `HYBRID_ALPHA`
- `CITATION_ALIGNMENT=true|false`
- `ALIGNMENT_MODE=disclaim|remove`
- `ANSWER_CACHE_TTL_SECONDS`
- `ANSWER_CACHE_MIN_SIMILARITY`
- `ANSWER_CACHE_STRICT_FINGERPRINT`
- `EVAL_MODE`
- `EVAL_SAMPLE_RATE`
- `EVAL_STORE_PATH`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `RAGAS_JUDGE_MODEL`
- `METRICS_MODE`
- `METRICS_STORE_PATH`

## Docker

Build:

```bash
docker build -t medical-llm-assistant:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env medical-llm-assistant:latest
```

The container healthcheck uses `GET /health`.

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`

It runs:

- `ruff check . --select E9,F63,F7,F82,F401`
- `pytest -q`
- `cd frontend && npm ci`
- `cd frontend && npm run build -- --configuration production`
- `cd frontend && npm test -- --watch=false --browsers=ChromeHeadless`

## Project Layout

- `api/`: FastAPI entrypoint, routers, API models, and session persistence.
- `frontend/`: Angular SPA, shared UI components, routing, and client-side state.
- `src/core/`: baseline retrieval and generation pipeline.
- `src/chat/`: follow-up handling and request routing.
- `src/integrations/`: PubMed, Chroma, and NVIDIA integrations.
- `src/papers/`: cached paper content and per-paper indexing helpers.
- `src/observability/`: tracing and metrics helpers.
- `eval/`: evaluation pipeline and dashboards.
- `tests/`: backend unit and smoke tests.
