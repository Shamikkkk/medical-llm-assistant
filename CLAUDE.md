# CLAUDE.md — Medical LLM Assistant: Full Reference

## Project Overview

A PubMed-backed biomedical conversational assistant. Given a clinical or medical question, it searches PubMed, retrieves abstracts, embeds them in ChromaDB, and synthesizes an evidence-based answer using an NVIDIA-hosted LLM. The system enforces biomedical scope, validates citations, and optionally validates answer claims with an NLI model.

---

## Repository Structure

```
src/
  chat/             # Top-level request routing and follow-up contextualization
  core/             # Pipeline, chains, retrieval, config, scope, intent
  agent/            # Agent mode: orchestrator, tools, state, config, runtime
  integrations/     # External APIs: NVIDIA LLM, PubMed, ChromaDB storage
  validators/       # NLI answer validation: claim splitter, premise builder, scoring
  papers/           # Per-paper full-text fetch, index, DOI utilities
  utils/            # Answer post-processing, text, export, loading
  observability/    # Metrics reader/summarizer, tracing stub
  intent.py         # Intent classification (medical vs. smalltalk)
  history.py        # In-memory chat session history
  types.py          # TypedDicts: SourceItem, PipelineResponse
  logging_utils.py  # Structured JSON event logging and LLM token logging
  validator.py      # NLI model loader (get_validator)
  # Thin re-export shims: pipeline.py, config.py, models.py, pubmed.py, storage.py, scope.py
eval/               # Offline/online evaluation: evaluator, RAGAS, datasets, store, dashboard
scripts/            # Smoke tests for agent and validator
tests/              # Unit/integration tests
data/               # eval_set.json, papers/cache/, chroma/, eval/, metrics/
```

---

## Execution Modes

### Pipeline Mode (default, `AGENT_MODE=false`)
Direct sequential execution in `src/core/pipeline.py`: `invoke_chat()` / `stream_chat()`.

### Agent Mode (`AGENT_MODE=true`)
Orchestrated in `src/agent/orchestrator.py`: `invoke_agent_chat()` / `stream_agent_chat()`.

- With `AGENT_USE_LANGGRAPH=true` (default): runs a LangGraph `StateGraph` with nodes: `guard → refine → search → retrieve → synthesize`.
- Falls back to sequential execution if LangGraph import fails.

Both modes share the same tool functions in `src/agent/tools.py`.

---

## End-to-End Request Flow

```
User Query
  |
  v
src/chat/router.py::invoke_chat_request()  [or stream_chat_request()]
  |
  +--> src/chat/contextualize.py::contextualize_question()
  |       Detects follow-up queries (short, pronoun-heavy, or explicit follow_up_mode).
  |       LLM rewrites to self-contained query; heuristic fallback appends prior context.
  |
  +--> src/agent/runtime.py::invoke_chat_with_mode()
         |
         +--[agent_mode=true]---> src/agent/orchestrator.py
         |                          Answer cache lookup (ChromaDB exact+similar match)
         |                          If cache hit: return cached payload
         |                          Otherwise: run LangGraph or sequential agent
         |
         +--[agent_mode=false]--> src/core/pipeline.py
                                    Answer cache lookup
                                    If cache hit: return cached payload
                                    Otherwise: run pipeline steps
```

### Step 1 — Safety Guardrail (`src/agent/tools.py::safety_guardrail_tool`)
1. `normalize_user_query()` — typo correction + lowercasing/normalization.
2. `classify_intent_details()` — classifies as `medical` or `smalltalk`.
   - **Override**: Smoking/tobacco/nicotine queries are always `medical`.
   - **LLM path**: asks LLM for one-token label; caches in LRU dict (512 entries).
   - **Heuristic fallback**: keyword match → short non-medical heuristic → default to `medical`.
   - Smalltalk short-circuits (confidence >= 0.8) with a canned response.
3. `classify_scope()` — checks if the query is biomedical.
   - Forced medical override (smoking terms) → allow.
   - Biomedical keyword/phrase match → allow.
   - History context (session's last 6 messages) + ambiguous follow-up check → allow.
   - LLM JSON classification (label: `BIOMEDICAL | MULTI_SYSTEM_OVERLAP | OUT_OF_SCOPE`) → route.
   - Non-biomedical word match → deny.
   - Medical morphology suffix match → allow.
   - Returns `ScopeResult(label, allow, user_message, reframed_query)`.
4. Personal medical advice check (regex patterns: "should I", "my symptoms", "diagnose me").

### Step 2 — Query Refinement (`src/agent/tools.py::query_refinement_tool`)
- If `scope.reframed_query` exists, use it directly.
- Otherwise, call `rewrite_to_pubmed_query()`: LLM converts natural language to PubMed-style query (MeSH terms, AND connectives, ≤12 words).
- Returns `pubmed_query` (for PubMed API) and `retrieval_query` (for vector store).

### Step 3 — PubMed Search (`src/agent/tools.py::pubmed_search_tool`)
1. Check query result cache (in-memory `QueryResultCache` LRU + ChromaDB persistent).
   - Cache hit with sufficient PMIDs: use cached PMIDs.
   - Cache hit but stale/insufficient: refresh with new search.
   - Cache miss: run full search.
2. If `MULTI_STRATEGY_RETRIEVAL=true`: generate 3 query variants (primary, MeSH-only, broad OR-based) via LLM.
3. `multi_strategy_esearch()`: parallel `pubmed_esearch()` calls via `ThreadPoolExecutor` (max 4 workers), merges deduplicated PMIDs.
4. `pubmed_efetch()`: fetches XML records from PubMed EFetch API; parses title, abstract, journal, year, authors, DOI, PMCID, fulltext_url.
5. `to_documents()`: converts records to LangChain `Document` objects.
6. `upsert_abstracts()`: embeds new documents (by PMID) into ChromaDB abstract store (skips already-present PMIDs).
7. `remember_query_result()`: persists PMIDs to in-memory + ChromaDB query cache.

### Step 4 — Retrieval (`src/agent/tools.py::retriever_tool`)
1. `abstract_store.as_retriever(k=candidate_k)` — vector similarity search.
   - `candidate_k = top_n * retrieval_candidate_multiplier` (default 3x, max 50).
2. Optional `FlashrankRerank` (`USE_RERANKER=true`) via `ContextualCompressionRetriever`.
3. `hybrid_rerank_documents()` — BM25 + semantic score weighted by `HYBRID_ALPHA` (0=BM25, 1=semantic).
4. `select_context_documents()` — deduplicates by Jaccard similarity (threshold 0.92), limits to `top_n`.

### Step 5 — Answer Synthesis (`src/agent/tools.py::answer_synthesis_tool`)
1. `build_rag_chain(llm, retriever)` — LangChain chain:
   - Retrieves documents via `retrieval_query`.
   - Formats context with `build_context_rows()` (token-budget-aware: distributes budget across docs, truncates or compresses per `CONTEXT_TRIM_STRATEGY`).
   - Fills `ChatPromptTemplate`: system prompt + chat history + `"Question: {input}\n\nAbstracts:\n{context}"`.
2. `build_chat_chain(base_chain)` — wraps with `RunnableWithMessageHistory` (keyed by `session_id` + `branch_id`).
3. Invokes (or streams) via NVIDIA ChatNVIDIA (temperature=0, streaming=True).

**System Prompt** (in `src/core/chains.py`):
- Forces structured output: `## Direct Answer`, `## Evidence Summary` (each bullet ends with `[PMID: XXXXXXXX]`), `## Evidence Quality` (`Strong / Moderate / Preliminary / Insufficient`), `## Caveats` (optional).
- Strict rules: never invent PMIDs, never say "I don't have access to full text", ≤450 words.

### Step 6 — Post-Processing
1. `annotate_answer_metadata()` (`src/utils/answers.py`):
   - `validate_citations_in_answer()`: replaces cited PMIDs not in sources with `[PMID: UNAVAILABLE]`; logs invalid citations.
   - `extract_evidence_quality()`: extracts quality label from `## Evidence Quality` section.
2. Optional `align_answer_citations()` (`src/core/retrieval.py`):
   - For each medical claim sentence (contains claim-hint keywords or numerics), checks token overlap with retrieved contexts.
   - Unsupported sentences: appended with `[No supporting PMID found...]` (mode=`disclaim`) or removed (mode=`remove`).
3. Optional NLI validation (`VALIDATOR_ENABLED=true`) via `src/validators::validate_answer()`:
   - Splits answer into claims (`claim_splitter.py`).
   - For each claim: selects top evidence chunks by Jaccard similarity, selects top sentences per chunk (`premise_builder.py`).
   - Scores claim vs. premise using DeBERTa MNLI model (`scoring.py::score_claim_with_nli()`).
   - Falls back to heuristic token-overlap scoring if model is not entailment-tuned or inference fails.
   - Aggregates: returns `valid` bool, `score`, `label`, `details` with per-claim breakdown.

### Step 7 — Answer Cache Storage
`store_answer_cache()` stores the response payload in ChromaDB `answer_cache` collection keyed by:
- Normalized query text (for exact lookup).
- Config fingerprint (SHA256 of model, reranker, abstracts, token budget, validator, top_n, backend settings).
- UTC timestamp for TTL enforcement.

---

## Key Files Reference

### `src/core/config.py`
- `AppConfig` — frozen dataclass with all configuration fields.
- `load_config()` — reads `.env` via `python-dotenv`, parses all env vars with validation and clamping.
- `ConfigValidationError` — raised by `AppConfig.require_valid()` for fatal config issues (e.g., `EVAL_MODE` requires `OPENAI_API_KEY`).

### `src/core/pipeline.py`
Full pipeline for non-agent mode. Mirrors agent tool sequence but as a single monolithic function. Handles: answer cache, scope, follow-up detection (`FOLLOWUP_PATTERNS`), multi-strategy PubMed fetch, embedding, RAG chain, citation alignment, validation, eval hook.

### `src/core/chains.py`
- `SYSTEM_PROMPT` — the clinical literature assistant prompt.
- `build_rag_chain(llm, retriever, max_abstracts, max_context_tokens, trim_strategy)`.
- `build_chat_chain(base_chain)` — wraps with session/branch history.
- `_format_docs()` — bridges retriever output to `build_context_text()`.

### `src/core/retrieval.py`
- `hybrid_rerank_documents(query, docs, alpha, limit)` — BM25 (TF-IDF with k1=1.5, b=0.75) + semantic (from metadata score or Jaccard fallback), normalized, linearly blended.
- `build_context_rows(docs, max_abstracts, max_context_tokens, trim_strategy)` — per-doc budget allocation.
- `build_context_text()` — joins rows with `\n\n---\n\n` separator.
- `deduplicate_by_semantic_similarity(docs, threshold=0.92)` — Jaccard on tokenized text.
- `align_answer_citations(answer, contexts, mode)` — sentence-level citation check.
- `count_tokens(text)` — tiktoken (cl100k_base) with char/4 fallback.

### `src/core/scope.py`
- `classify_scope(query, session_id, llm)` — full scope classification with fallback chain.
- `ScopeResult(label, allow, user_message, reframed_query, reason)` — dataclass.
- `BIOMEDICAL_PHRASES`, `BIOMEDICAL_WORDS`, `NON_BIOMEDICAL_WORDS`, `SYSTEM_KEYWORDS` — heuristic keyword sets.

### `src/intent.py`
- `classify_intent_details(user_text, llm, log_enabled)` — returns `{label, confidence, reason}`.
- `normalize_user_query(text)` — typo correction + normalize whitespace/case.
- `correct_common_medical_typos(text)` — token-level corrections + difflib fuzzy match for smoking terms.
- `is_forced_medical_query(text)` — override for smoking/tobacco domain.
- `smalltalk_reply(query)` — deterministic smalltalk responses.
- `should_short_circuit_smalltalk(intent_details, user_text)` — blocks retrieval if confidence >= 0.8.
- LLM result cache: `OrderedDict` LRU (512 entries).

### `src/integrations/nvidia.py`
- `get_nvidia_llm(model_name, api_key)` — returns `ChatNVIDIA` via `lru_cache`.
- Default model: `meta/llama-3.1-8b-instruct`. Temperature=0, streaming=True.

### `src/integrations/pubmed.py`
- `pubmed_esearch(term, retmax)` — NCBI ESearch JSON API.
- `pubmed_efetch(pmids)` — NCBI EFetch XML API; extracts PMID, title, abstract (multi-section), journal, year, authors, DOI, PMCID, fulltext_url.
- `build_multi_strategy_queries(user_query, llm)` — primary + MeSH + broad variants.
- `multi_strategy_esearch(queries, retmax_each)` — parallel ThreadPoolExecutor, deduplicates PMIDs, cap at `retmax_each * 2`.
- `rewrite_to_pubmed_query(user_query, llm)` — LLM rewrite prompt; falls back to normalized query.
- `to_documents(records)` — `Document(page_content=title+abstract, metadata={pmid, title, journal, year, authors, doi, pmcid, fulltext_url})`.
- Rate limiting: 0.34s sleep between requests; timeout 20s.

### `src/integrations/storage.py`
Three ChromaDB collections in `{DATA_DIR}/chroma/`:
- `query_cache` — stores PubMed query results (PMIDs) keyed by normalized query text.
- `pubmed_abstracts` — embedded abstract documents, keyed by PMID.
- `answer_cache` — full response payloads keyed by query + config fingerprint.

Key classes/functions:
- `QueryResultCache` — in-memory `OrderedDict` LRU with TTL and negative-cache support.
- `lookup_query_result_cache()` — checks in-memory first, then ChromaDB persistent.
- `remember_query_result()` — writes to in-memory + ChromaDB (skips empty PMID lists).
- `lookup_answer_cache()` — exact match (ChromaDB `where` filter) then similarity match.
- `store_answer_cache()` — stores payload JSON as metadata field; only for `status="answered"`.
- `build_answer_cache_fingerprint()` — SHA256 of config dict for cache isolation.
- `upsert_abstracts()` — checks existing IDs, only embeds new documents.
- `get_embeddings()` — `SentenceTransformerEmbeddings("all-MiniLM-L6-v2")`, cached by `lru_cache`.
- Stores are built via `lru_cache` on (persist_dir, collection_name, embeddings_model, device).

### `src/agent/orchestrator.py`
- `invoke_agent_chat()` / `stream_agent_chat()` — public entry points.
- `_run_agent()` — tries LangGraph runner, falls back to `_run_sequential()`.
- `_get_langgraph_runner()` — `lru_cache(maxsize=1)`, builds and compiles graph once.
- LangGraph graph edges: `guard --[allowed]--> refine --> search --> retrieve --> synthesize --> END`; `guard --[blocked]--> END`.
- `_state_to_payload()` — converts `AgentState` to `PipelineResponse`.
- `_run_optional_validation()` — calls `validate_answer()` if `VALIDATOR_ENABLED`.

### `src/agent/tools.py`
All stateless tool functions. Each takes explicit parameters and returns a dict.
- `safety_guardrail_tool(query, session_id, llm, log_pipeline)`.
- `query_refinement_tool(query, scope, llm)`.
- `pubmed_search_tool(query, pubmed_query, top_n, persist_dir, log_pipeline, compute_device)`.
- `retriever_tool(abstract_store, retrieval_query, top_n, use_reranker, log_pipeline)`.
- `answer_synthesis_tool(query, retrieval_query, session_id, branch_id, llm, retriever, context_top_k)`.
- `answer_synthesis_stream_tool(...)` — generator; `yield` text chunks, `return` final dict.
- `citation_formatting_tool(docs, top_n)` → `list[SourceItem]`.
- `context_export_tool(docs, top_n)` → `list[dict[str, str]]`.

### `src/agent/state.py`
`AgentState(TypedDict, total=False)` — full state schema passed between LangGraph nodes.

### `src/agent/runtime.py`
`invoke_chat_with_mode()` / `stream_chat_with_mode()` — dispatches to agent or pipeline based on `agent_mode` flag. Lazy imports to avoid circular dependencies.

### `src/chat/router.py`
`invoke_chat_request()` / `stream_chat_request()` — UI-facing handlers:
- Generates `request_id` (UUID hex).
- Contextualizes query.
- Logs `request.start` event.
- Calls `invoke_chat_with_mode()` or `stream_chat_with_mode()`.
- Drops legacy `paper_*` fields from payload.
- Adds `effective_query`, `rewritten_query`, `last_topic_summary`, `branch_id`.

### `src/chat/contextualize.py`
`contextualize_question(user_query, chat_messages, follow_up_mode, conversation_summary, llm)`:
- Returns `(effective_query, topic_summary, rewritten: bool)`.
- Triggers if `follow_up_mode=True` OR query has ≤7 tokens OR contains pronouns/follow-up keywords.
- LLM path: fills `REWRITE_PROMPT` with topic_summary + history + query; returns first 25-word cleaned sentence.
- Heuristic fallback: appends `"in the context of: {context_hint}"`.

### `src/history.py`
- In-memory `dict[(session_id, branch_id)] → BaseChatMessageHistory`.
- `get_session_history(session_id, branch_id)` — creates if not present.
- `replace_session_history()` — reconstructs from message list.
- `clear_session_history()` — clears one branch or all branches for a session.

### `src/types.py`
- `SourceItem` — `{rank, pmid, title, journal, year, doi, pmcid, fulltext_url, context}`.
- `PipelineResponse` — full response dict schema (all optional fields).

### `src/validators/gpt_oss_validator.py` (`validate_answer`)
Main validator orchestrator:
1. Extract evidence chunks from `retrieved_docs` or `context` string.
2. Split answer into claims via `split_into_claims()`.
3. Load NLI components via `get_nli_components()`.
4. For each claim: `build_claim_premise()` → `score_claim_with_nli()` (or heuristic fallback).
5. `aggregate_claim_scores()` → `{valid, score, label, details}`.
6. Penalizes score if answer cites PMIDs not in `source_pmids`.

### `src/validators/claim_splitter.py`
`split_into_claims(answer, max_claims=12)`:
- Strips bullet prefixes, splits by sentence boundary regex, normalizes whitespace.

### `src/validators/premise_builder.py`
`build_claim_premise(claim, evidence_chunks, tokenizer, top_n_chunks, top_k_sentences, max_premise_tokens)`:
- Ranks chunks by Jaccard similarity to claim; selects top N.
- Within each chunk, ranks sentences; selects top K.
- Packs sentences into token budget; truncates if needed.

### `src/validators/scoring.py`
`score_claim_with_nli(claim, premise, tokenizer, model, label_map, ...)`:
- Tokenizes premise+claim, runs model forward pass, softmax.
- Resolves contradiction/neutral/entailment label indices dynamically.
- `margin = entailment - contradiction`; normalized score `= (margin + 1) / 2`.
- Critical claims (numeric values or high-stakes keywords) use stricter thresholds.

`aggregate_claim_scores(claim_scores, margin, contradiction_limit)`:
- Valid if `avg_margin >= margin`, `max_contradiction <= contradiction_limit`, `critical_failures == 0`.

### `src/validator.py` (model loader)
`get_validator(model_name)`:
- Tries to auto-upgrade base `microsoft/deberta-v3-base` to locally cached MNLI variant.
- Loads `transformers.pipeline("text-classification", model=..., device=-1)`.
- Caches loaded pipelines in `_VALIDATOR_PIPES` dict.
- Returns `{pipeline, model_name, requested_model_name, from_cache}` or `None` if load fails.

### `src/papers/`
- `store.py`: `PaperContent` dataclass `{pmid, doi, fulltext_url, title, authors, year, journal, pubmed_url, abstract, full_text, content_tier, source_label, fetched_at, pmcid, notes}`. `PAPER_TIER_ABSTRACT = "abstract"`, `PAPER_TIER_FULL_TEXT = "full_text"`. `build_pubmed_url(pmid)`.
- `fetch.py`: `fetch_paper_content(pmid)` — fetches abstract via `pubmed_efetch`, tries PMC full text via `fetch_pmc_full_text()`. `extract_text_from_uploaded_pdf()` for PDF uploads (requires `pypdf`).
- `index.py`: `PaperIndexer` — per-paper ChromaDB collection (`paper_{pmid}`); chunks text (1800 chars, 300 overlap via `RecursiveCharacterTextSplitter`); skips already-indexed source hashes.
- `doi.py`: `extract_doi(raw)` — regex extraction of DOI pattern. `build_doi_url(doi)` — `https://doi.org/{doi}`.

### `src/logging_utils.py`
- `log_event(name, **fields)` — JSON log line to logger + optional append to JSONL file.
- `log_llm_usage(tag, response)` — extracts and logs token usage from LLM response metadata.
- `hash_query_text(text)` — SHA256 of normalized query (for privacy-safe logging).
- `extract_usage_stats(response)` — walks response metadata tree for prompt/completion/total tokens.

### `src/observability/metrics.py`
- `read_metric_events(path)` — reads JSONL metrics file.
- `summarize_metric_events(events)` — computes `total_requests`, `error_count`, `error_rate`, `latency_p50_ms`, `latency_p95_ms`, `cache_hit_rate`, `avg_pmid_count`.

### `eval/evaluator.py`
`evaluate_turn(query, answer, contexts, sources, expected_pmids, mode)`:
- Tries RAGAS scores; falls back to heuristic metrics.
- Heuristic metrics: `faithfulness` (context coverage), `answer_relevance`, `context_precision`, `context_recall`, `hallucination_risk`.
- `citation_metrics`: `citation_presence`, `citation_alignment` (fraction of cited PMIDs in sources).
- `safety_metrics`: detects personal medical advice requests; checks for disclaimer in response.
- `retrieval_metrics`: `recall_at_k`, `mrr` (mean reciprocal rank) if `expected_pmids` provided.
- Pass thresholds: faithfulness ≥ 0.6, answer_relevance ≥ 0.6, context_precision ≥ 0.45, context_recall ≥ 0.45, citation_alignment ≥ 0.8, safety_compliance ≥ 0.9.

---

## Configuration Reference

All settings read from environment variables (or `.env` file via `python-dotenv`).

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | *(required)* | NVIDIA API key for LLM |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | LLM model ID |
| `DATA_DIR` | `./data` | Root data/chroma directory |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `AGENT_MODE` | `false` | Enable agent orchestration |
| `AGENT_USE_LANGGRAPH` | `true` | Use LangGraph in agent mode |
| `USE_RERANKER` | `false` | Enable FlashrankRerank |
| `LOG_PIPELINE` | `false` | Verbose pipeline debug logging |
| `VALIDATOR_ENABLED` | `false` | Enable NLI answer validation |
| `VALIDATOR_MODEL_NAME` | `MoritzLaurer/DeBERTa-v3-base-mnli` | NLI model |
| `VALIDATOR_THRESHOLD` | `0.7` | Validation score threshold [0,1] |
| `VALIDATOR_MARGIN` | `0.2` | Entailment-contradiction margin |
| `VALIDATOR_MAX_PREMISE_TOKENS` | `384` | Max tokens for premise [64,1024] |
| `VALIDATOR_MAX_HYPOTHESIS_TOKENS` | `128` | Max tokens for hypothesis [32,512] |
| `VALIDATOR_MAX_LENGTH` | `512` | Max total NLI input length [128,1024] |
| `VALIDATOR_TOP_N_CHUNKS` | `4` | Top evidence chunks per claim [1,10] |
| `VALIDATOR_TOP_K_SENTENCES` | `2` | Top sentences per chunk [1,5] |
| `MAX_CONTEXT_ABSTRACTS` | `8` | Abstracts included in prompt [1,20] |
| `MAX_CONTEXT_TOKENS` | `2500` | Token budget for context [256,20000] |
| `CONTEXT_TRIM_STRATEGY` | `truncate` | `truncate` or `compress` (2-sentence) |
| `MULTI_STRATEGY_RETRIEVAL` | `true` | Generate 3 PubMed query variants |
| `RETRIEVAL_CANDIDATE_MULTIPLIER` | `3` | Fetch N×top_n candidates [1,10] |
| `HYBRID_RETRIEVAL` | `false` | BM25 + semantic hybrid reranking |
| `HYBRID_ALPHA` | `0.5` | Blend ratio (0=BM25, 1=semantic) [0,1] |
| `CITATION_ALIGNMENT` | `true` | Post-hoc citation validation |
| `ALIGNMENT_MODE` | `disclaim` | `disclaim` or `remove` unsupported sentences |
| `SHOW_REWRITTEN_QUERY` | `false` | Surface rewritten query in UI |
| `AUTO_SCROLL` | `true` | UI auto-scroll preference |
| `ANSWER_CACHE_TTL_SECONDS` | `604800` | Answer cache TTL (7 days) |
| `ANSWER_CACHE_MIN_SIMILARITY` | `0.9` | Minimum similarity for cache hit [0,1] |
| `ANSWER_CACHE_STRICT_FINGERPRINT` | `true` | Require exact config fingerprint match |
| `PUBMED_CACHE_TTL_SECONDS` | `604800` | PubMed query cache TTL |
| `PUBMED_NEGATIVE_CACHE_TTL_SECONDS` | `3600` | Negative (empty) cache TTL |
| `EVAL_MODE` | `false` | Enable evaluation hooks (requires OPENAI_API_KEY + OPENAI_BASE_URL) |
| `EVAL_SAMPLE_RATE` | `0.25` | Fraction of requests to evaluate [0,1] |
| `EVAL_STORE_PATH` | `./data/eval/eval_results.jsonl` | Eval output path |
| `METRICS_MODE` | `false` | Log events to JSONL file |
| `METRICS_STORE_PATH` | `./data/metrics/events.jsonl` | Metrics output path |

---

## Data Storage Layout

```
data/
  chroma/
    query_cache/        ChromaDB: PubMed query result cache (normalized query → PMIDs)
    pubmed_abstracts/   ChromaDB: Embedded PubMed abstracts (PMID → Document)
    answer_cache/       ChromaDB: Cached LLM responses (query+fingerprint → payload JSON)
  papers/
    chroma/             ChromaDB: Per-paper full-text indexes (paper_{pmid} collection)
    cache/              Raw JSON files for fetched papers
  eval/
    eval_results.jsonl  Online/offline evaluation records
  metrics/
    events.jsonl        Structured observability events (request.start, request.complete)
```

---

## Module Import Aliases (re-export shims)

These top-level files in `src/` are thin re-exports — the real code lives in submodules:

| Shim | Real implementation |
|---|---|
| `src/pipeline.py` | `src/core/pipeline.py` |
| `src/config.py` | `src/core/config.py` |
| `src/models.py` | `src/integrations/nvidia.py` |
| `src/pubmed.py` | `src/integrations/pubmed.py` |
| `src/storage.py` | `src/integrations/storage.py` |
| `src/scope.py` | `src/core/scope.py` |
| `src/core/intent.py` | `src/intent.py` |
| `src/core/smalltalk.py` | `src/intent.py` (classify_intent, smalltalk_reply) |
| `src/validators/__init__.py` | `src/validators/gpt_oss_validator.py` |

---

## Key Design Decisions

- **Two modes, one tool layer**: Pipeline and agent mode share `src/agent/tools.py` functions; pipeline mode calls them sequentially inline, agent mode dispatches via LangGraph nodes or sequential runner.
- **Three-layer answer cache**: In-memory LRU → ChromaDB exact match → ChromaDB similarity match. Cache isolated by SHA256 config fingerprint.
- **Two-layer query cache**: In-memory LRU (`QueryResultCache`) → ChromaDB persistent. Negative (empty-result) queries cached with shorter TTL.
- **Hybrid retrieval**: BM25 (custom implementation, no external library) + semantic score from Chroma metadata, linearly blended.
- **NLI validator is optional and non-blocking**: If model fails to load, validator returns `valid=True`. Heuristic fallback used if model is not MNLI-tuned.
- **Chat branching**: History keyed by `(session_id, branch_id)` tuple, allowing multiple conversation branches per session.
- **Smoking/tobacco override**: These queries are hard-coded as biomedical to prevent misclassification as smalltalk or out-of-scope.
- **No streaming in pipeline mode without agent**: `stream_chat()` in pipeline mode yields tokens via LangChain `.stream()`.
- **Papers submodule is standalone**: `PaperIndexer` maintains per-PMID Chroma collections for deep follow-up Q&A on individual papers; separate from the main abstract store.
