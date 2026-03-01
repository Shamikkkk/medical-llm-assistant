# 🧬 PubMed Literature Assistant

A full-stack biomedical literature Q&A application powered by PubMed abstracts, NVIDIA-hosted LLMs, and a modern Angular interface.

## Features

- 🔍 **Multi-strategy PubMed retrieval** with BM25 hybrid reranking
- 🤖 **NVIDIA-hosted LLM answer generation** with citation validation and evidence quality scoring
- 💬 **Claude-style branch navigation** — edit any message to fork the conversation
- 🌗 **Light/dark mode** toggle with localStorage persistence
- ⚡ **Similar-query answer cache** to avoid redundant LLM calls
- 📊 **Optional evaluation and metrics pipelines** via RAGAS
- 🐳 **Docker-ready** with a single build command

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Angular 17, Tailwind CSS |
| Backend | FastAPI, SSE streaming |
| LLM | NVIDIA-hosted chat models (LangChain) |
| Retrieval | Chroma, SentenceTransformers, PubMed E-utilities |
| Evaluation | RAGAS, DeBERTa NLI validator |

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+ — https://nodejs.org
- An NVIDIA API key — https://build.nvidia.com (free tier available)

### 1. Clone the repository
```bash
git clone https://github.com/Shamikkkk/medical-llm-assistant.git
cd medical-llm-assistant
```

### 2. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and set your API key:
```
NVIDIA_API_KEY=your_key_here
```

### 3. Set up the Python backend
```bash
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Set up the Angular frontend
```bash
npm install -g @angular/cli@17
cd frontend
npm ci
cd ..
```

### 5. Run the application

Open **two terminals** from the project root:

**Terminal 1 — Backend:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm start
```

Then open **http://localhost:4200** in your browser.

> The Angular dev server proxies all `/api` calls to FastAPI on port 8000 automatically.

---

## Docker
```bash
docker build -t medical-llm-assistant:latest .
docker run --rm -p 8000:8000 --env-file .env medical-llm-assistant:latest
```

Then open **http://localhost:8000** in your browser.

---

## Tests

**Backend:**
```bash
pytest -q
```

**Frontend:**
```bash
cd frontend
npm test -- --watch=false --browsers=ChromeHeadless
```

**Network smoke tests (opt-in):**
```bash
# Windows
$env:RUN_NETWORK_TESTS="true"
pytest -q tests/test_retrieval_smoke.py

# Mac/Linux
RUN_NETWORK_TESTS=true pytest -q tests/test_retrieval_smoke.py
```

---

## Project Layout
```
api/            FastAPI entrypoint, routers, session persistence
frontend/       Angular SPA — components, services, routing
src/core/       Retrieval and generation pipeline
src/chat/       Follow-up handling and request routing
src/integrations/  PubMed, Chroma, and NVIDIA integrations
src/papers/     Cached paper content and per-paper indexing
src/observability/ Tracing and metrics helpers
src/validators/ NLI-based answer support validation
eval/           Evaluation pipeline and dashboards
tests/          Backend unit and smoke tests
```

---

## Key Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `NVIDIA_API_KEY` | Required for LLM answer generation | — |
| `NVIDIA_MODEL` | Generation model name | `meta/llama-3.1-8b-instruct` |
| `DATA_DIR` | Root directory for Chroma and local data | `./data` |
| `MAX_CONTEXT_ABSTRACTS` | Max abstracts passed to LLM context | `8` |
| `MAX_CONTEXT_TOKENS` | Token budget for context window | `2500` |
| `HYBRID_RETRIEVAL` | Enable BM25 + semantic hybrid reranking | `false` |
| `MULTI_STRATEGY_RETRIEVAL` | Enable parallel multi-query PubMed search | `true` |
| `ANSWER_CACHE_TTL_SECONDS` | Answer cache expiry | `604800` |
| `EVAL_MODE` | Enable online RAGAS evaluation | `false` |
| `METRICS_MODE` | Enable metrics dashboard | `false` |
| `AGENT_MODE` | Use LangGraph agent instead of pipeline | `false` |

See `.env.example` for the full list of configuration options.

---

## CI

GitHub Actions runs on every push and pull request:
- `ruff check` (lint)
- `pytest -q` (backend tests)
- `ng build --configuration production` (frontend build)
- `ng test` (frontend tests)