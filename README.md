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
- Similar-query answer caching with config fingerprinting and TTL checks.
- Separate display Top-N, retrieval fetch budget, and prompt-context budgeting.
- Context budgeting with `MAX_CONTEXT_ABSTRACTS` and `MAX_CONTEXT_TOKENS`.
- Optional hybrid retrieval with lexical + semantic fusion.
- Follow-up rewriting using rolling conversation summary memory.
- Conversation branching from any prior user prompt.
- CPU / GPU selector for local embeddings and the optional validator.
- Citation alignment disclaimers for unsupported answer sentences.
- Structured JSON request logs and optional metrics dashboard.
- Streamed answers with topic-aware thinking status, auto-scroll, and per-answer copy button.
- Sidebar export, cache-clearing controls, latency panel, and source inspector.

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
- `MAX_CONTEXT_ABSTRACTS`
- `MAX_ABSTRACTS` (deprecated alias for `MAX_CONTEXT_ABSTRACTS`)
- `MAX_CONTEXT_TOKENS`
- `CONTEXT_TRIM_STRATEGY=truncate|compress`
- `HYBRID_RETRIEVAL=true|false`
- `HYBRID_ALPHA`
- `CITATION_ALIGNMENT=true|false`
- `ALIGNMENT_MODE=disclaim|remove`

UI:

- `SHOW_REWRITTEN_QUERY`
- `AUTO_SCROLL`
- The app locks a fixed dark Streamlit theme via `.streamlit/config.toml`

Similar-answer cache:

- `ANSWER_CACHE_TTL_SECONDS`
- `ANSWER_CACHE_MIN_SIMILARITY`
- `ANSWER_CACHE_STRICT_FINGERPRINT`

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
- The sidebar controls expose `Top-N papers`, `Follow-up mode`, `Show papers`, `Show rewritten query`, `Auto-scroll`, and `Compute device`.
- `Top-N papers` controls how many unique papers the app attempts to display in `docs_preview` and ranked sources, up to 10.
- Candidate PubMed fetch size is derived from the requested `Top-N papers` value and expands automatically when reranking or hybrid retrieval is enabled.
- `MAX_CONTEXT_ABSTRACTS` controls how many abstracts are fed into the answer prompt and validator context budget.
- `MAX_ABSTRACTS` is kept as a deprecated backward-compatible alias for `MAX_CONTEXT_ABSTRACTS`.
- `Show papers` only affects paper-link rendering. Retrieval and PMID-grounded answering still run when it is off.
- `Compute device` affects only local embeddings and the optional validator. Remote NVIDIA answer generation is unchanged.
- `Compute device = Auto` uses GPU when CUDA is available and otherwise falls back to CPU.
- If PubMed returns fewer unique papers than the requested Top-N, the app shows all available unique papers and adds a short note in the response.
- Smoking-cessation questions, including common typos like `quti smoking`, are routed through biomedical retrieval instead of smalltalk fallback.
- Editing a previous user prompt creates a new branch that keeps the original branch intact.
- Cached answers are reused only when the answer-cache TTL is still valid and the runtime fingerprint matches, unless `ANSWER_CACHE_STRICT_FINGERPRINT=false`.
- Cached answers render with a small “Cached answer (similar query)” badge and timestamp metadata.

## Branching and Sidebar Tools

- Use `Edit prompt` under any prior user message to create a new branch from that point.
- Use the sidebar branch selector to switch between branches inside the same chat tree.
- Use the export controls to download the active branch as Markdown or JSON.
- Use the cache maintenance controls to clear query cache, answer cache, or paper cache artifacts.
- Use the latency panel to inspect the latest request timings.
- Use the source inspector under each rendered source to view the abstract context used for the answer.

## Docker

Build:

```bash
docker build -t medical-llm-assistant:latest .
```

Run:

```bash
docker run --rm --gpus all -p 8501:8501 --env-file .env medical-llm-assistant:latest
```

The container healthcheck uses Streamlit's `/_stcore/health` endpoint.

GPU is optional. The default Dockerfile remains CPU-friendly. To use GPU in Docker, run with NVIDIA Container Toolkit on a compatible host; this repository does not switch the base image automatically.

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
