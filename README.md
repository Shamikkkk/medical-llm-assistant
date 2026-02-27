# PubMed Literature Assistant

Streamlit application for biomedical literature Q&A grounded in PubMed abstracts.

## Local Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from the example file:

```bash
cp .env.example .env
```

4. Set `NVIDIA_API_KEY` if you want answer generation enabled.
5. Start the app:

```bash
streamlit run app.py
```

## Tests

```bash
pytest -q
```

Network smoke tests are opt-in:

```bash
$env:RUN_NETWORK_TESTS="true"
pytest -q tests/test_retrieval_smoke.py
```

## Core Features

- Process-level caching for embeddings, Chroma stores, and the NVIDIA chat client.
- PubMed query TTL caching with negative-cache handling for empty search results.
- Context budgeting with `MAX_ABSTRACTS` and `MAX_CONTEXT_TOKENS`.
- Optional hybrid retrieval with lexical + semantic fusion.
- Follow-up rewriting using rolling conversation summary memory.
- Citation alignment disclaimers for unsupported answer sentences.
- Structured JSON request logs and optional metrics dashboard.
- Streamed answers with topic-aware thinking status, auto-scroll, and per-answer copy button.

## Key Configuration

Core:

- `NVIDIA_API_KEY`: required for LLM answer generation.
- `NVIDIA_MODEL`: generation model name.
- `DATA_DIR`: root directory for Chroma persistence and local data.
- `USE_RERANKER`: enables Flashrank reranking when available.
- `LOG_PIPELINE`: enables detailed pipeline logs.

Retrieval and context:

- `PUBMED_CACHE_TTL_SECONDS`
- `PUBMED_NEGATIVE_CACHE_TTL_SECONDS`
- `MAX_ABSTRACTS`
- `MAX_CONTEXT_TOKENS`
- `CONTEXT_TRIM_STRATEGY=truncate|compress`
- `HYBRID_RETRIEVAL=true|false`
- `HYBRID_ALPHA`
- `CITATION_ALIGNMENT=true|false`
- `ALIGNMENT_MODE=disclaim|remove`

UI:

- `SHOW_REWRITTEN_QUERY`
- `AUTO_SCROLL`

Optional evaluation and metrics:

- `EVAL_MODE`
- `EVAL_SAMPLE_RATE`
- `EVAL_STORE_PATH`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `RAGAS_JUDGE_MODEL`
- `METRICS_MODE`
- `METRICS_STORE_PATH`

## Runtime Notes

- If `NVIDIA_API_KEY` is missing, the app still performs retrieval but returns a configuration message instead of a generated answer.
- If `EVAL_MODE=true`, startup validation requires `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- The sidebar controls expose `Top-N papers`, `Follow-up mode`, `Show papers`, `Show rewritten query`, and `Auto-scroll`.
- `Show papers` only affects paper-link rendering. Retrieval and PMID-grounded answering still run when it is off.

## Docker

Build:

```bash
docker build -t medical-llm-assistant .
```

Run:

```bash
docker run --rm -p 8501:8501 --env-file .env medical-llm-assistant
```

The container healthcheck uses Streamlit's `/_stcore/health` endpoint.

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`

It runs:

- `ruff check . --select E9,F63,F7,F82,F401`
- `pytest -q`

## Project Layout

- `app.py`: Streamlit entrypoint.
- `src/core/`: baseline retrieval and generation pipeline.
- `src/chat/`: follow-up handling and request routing.
- `src/integrations/`: PubMed, Chroma, and NVIDIA integrations.
- `src/ui/`: Streamlit rendering helpers.
- `src/observability/`: tracing and metrics helpers.
- `eval/`: evaluation pipeline and dashboards.
- `tests/`: unit and smoke tests.
