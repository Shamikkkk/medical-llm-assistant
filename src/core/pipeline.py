from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from time import perf_counter
from typing import Any, Dict, List, Mapping
from uuid import uuid4
import logging
import re

from src.core.chains import _format_docs, build_chat_chain, build_rag_chain
from src.core.config import AppConfig, load_config
from src.core.retrieval import (
    align_answer_citations,
    build_context_rows,
    hybrid_rerank_documents,
    select_context_documents,
)
from src.core.scope import ScopeResult, classify_scope
from src.history import get_session_history
from src.intent import (
    classify_intent_details,
    normalize_user_query,
    should_short_circuit_smalltalk,
    smalltalk_reply,
)
from src.integrations.nvidia import get_nvidia_llm
from src.integrations.pubmed import (
    build_multi_strategy_queries,
    multi_strategy_esearch,
    pubmed_efetch,
    pubmed_esearch,
    rewrite_to_pubmed_query,
    to_documents,
)
from src.integrations.storage import (
    build_answer_cache_fingerprint,
    get_abstract_store,
    get_answer_cache_store,
    get_query_cache_store,
    lookup_answer_cache,
    lookup_query_result_cache,
    remember_query_result,
    store_answer_cache,
    upsert_abstracts,
)
from src.logging_utils import hash_query_text, log_event, log_llm_usage
from src.observability.tracing import start_span
from src.types import PipelineResponse, SourceItem
from src.utils.answers import annotate_answer_metadata
from src.validators import validate_answer

LOGGER = logging.getLogger("pipeline.main")

FOLLOWUP_PATTERNS = (
    "tell me more",
    "more about",
    "what about",
    "what else",
    "follow up",
    "follow-up",
    "expand on",
    "elaborate",
    "go deeper",
    "dosage",
    "dose",
    "safety",
    "side effects",
    "adverse",
    "tolerability",
)

PRONOUN_PATTERN = re.compile(
    r"\b(it|they|this|that|those|these|them|its|their|here|there)\b",
    flags=re.IGNORECASE,
)
QUERY_CACHE_PMID_STORE_LIMIT = 50


class _PipelineRetriever:
    def __init__(
        self,
        *,
        abstract_store: Any,
        candidate_k: int,
        final_k: int,
        compressor: Any | None,
        hybrid_retrieval: bool,
        hybrid_alpha: float,
        log_pipeline: bool,
    ) -> None:
        self._base_retriever = abstract_store.as_retriever(search_kwargs={"k": candidate_k})
        self._compressor = compressor
        self._hybrid_retrieval = bool(hybrid_retrieval)
        self._hybrid_alpha = float(hybrid_alpha)
        self._final_k = max(1, int(final_k))
        self._log_pipeline = bool(log_pipeline)

    def invoke(self, retrieval_query: str) -> list:
        with start_span(
            "chroma.query",
            attributes={"retrieval_query_hash": hash_query_text(retrieval_query)},
        ):
            try:
                docs = self._base_retriever.invoke(retrieval_query)
            except Exception:
                docs = []
        docs_list = list(docs or []) if not isinstance(docs, list) else docs

        if self._compressor is not None and docs_list:
            with start_span("rerank", attributes={"stage": "flashrank"}):
                try:
                    docs_list = list(
                        self._compressor.compress_documents(docs_list, query=retrieval_query)
                    )
                except Exception:
                    docs_list = docs_list

        if docs_list and len(docs_list) > self._final_k:
            with start_span("rerank", attributes={"stage": "hybrid"}):
                docs_list = hybrid_rerank_documents(
                    retrieval_query,
                    docs_list,
                    alpha=self._hybrid_alpha,
                    limit=len(docs_list),
                    explain_reranking=self._log_pipeline,
                )
        docs_list = select_context_documents(docs_list, max_abstracts=self._final_k)

        _pipeline_log(
            self._log_pipeline,
            "[PIPELINE] Retriever invoke complete | returned_docs_len=%s unique_pmids=%s hybrid=%s",
            len(docs_list),
            _count_unique_pmids_from_docs(docs_list),
            self._hybrid_retrieval,
        )
        return docs_list

    def get_relevant_documents(self, query: str) -> list:
        return self.invoke(query)


def run_pipeline(query: str) -> PipelineResponse:
    """Backwards-compatible wrapper for single-turn usage."""
    return invoke_chat(query, session_id="default", top_n=10)


def invoke_chat(
    query: str,
    session_id: str,
    branch_id: str = "main",
    top_n: int = 10,
    *,
    request_id: str | None = None,
    include_paper_links: bool = True,
    compute_device: str | None = None,
) -> PipelineResponse:
    config = load_config()
    current_request_id = request_id or uuid4().hex
    start_time = perf_counter()
    try:
        return _invoke_chat_impl(
            query=query,
            session_id=session_id,
            branch_id=branch_id,
            top_n=top_n,
            config=config,
            request_id=current_request_id,
            include_paper_links=include_paper_links,
            start_time=start_time,
            compute_device=compute_device,
        )
    except Exception as exc:
        _log_request_error(
            config=config,
            request_id=current_request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            started_at=start_time,
            error=exc,
        )
        raise


def stream_chat(
    query: str,
    session_id: str,
    branch_id: str = "main",
    top_n: int = 10,
    *,
    request_id: str | None = None,
    include_paper_links: bool = True,
    compute_device: str | None = None,
):
    config = load_config()
    current_request_id = request_id or uuid4().hex
    start_time = perf_counter()
    try:
        return (yield from _stream_chat_impl(
            query=query,
            session_id=session_id,
            branch_id=branch_id,
            top_n=top_n,
            config=config,
            request_id=current_request_id,
            include_paper_links=include_paper_links,
            start_time=start_time,
            compute_device=compute_device,
        ))
    except Exception as exc:
        _log_request_error(
            config=config,
            request_id=current_request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            started_at=start_time,
            error=exc,
        )
        raise


def build_contextual_retrieval_query(
    user_query: str,
    session_id: str,
    branch_id: str = "main",
    llm: Any | None = None,
    base_query: str | None = None,
) -> str:
    base_query = base_query or user_query
    normalized = " ".join(user_query.lower().strip().split())
    if not normalized:
        return base_query

    if not _is_followup_query(normalized):
        return base_query

    history_excerpt = _get_history_excerpt(session_id, branch_id=branch_id)
    if not history_excerpt:
        return base_query

    if llm is not None:
        prompt = (
            "Rewrite the follow-up question into a standalone PubMed retrieval query "
            "in the biomedical context. Use history to resolve references. "
            "Return ONLY the rewritten query string.\n"
            f"History: {history_excerpt}\n"
            f"Follow-up: {user_query}\n"
        )
        rewritten = _invoke_llm(llm, prompt, usage_tag="contextual_rewrite")
        if rewritten:
            return _sanitize_query(rewritten, fallback=base_query)

    snippet = _get_last_assistant_excerpt(session_id, branch_id=branch_id)
    if snippet:
        expanded = f"{base_query} {snippet}"
        return expanded[:300].strip()
    return base_query


def _invoke_chat_impl(
    *,
    query: str,
    session_id: str,
    branch_id: str = "main",
    top_n: int,
    config: AppConfig,
    request_id: str,
    include_paper_links: bool,
    start_time: float,
    compute_device: str | None = None,
) -> PipelineResponse:
    requested_top_n = _sanitize_top_n(top_n)
    normalized_query = normalize_user_query(query)
    llm = _get_llm_safe(config)
    intent = classify_intent_details(normalized_query, llm, log_enabled=config.log_pipeline)
    intent_label = str(intent.get("label", "")).lower()
    intent_confidence = float(intent.get("confidence", 0.0) or 0.0)
    if should_short_circuit_smalltalk(intent, normalized_query):
        reply = smalltalk_reply(query, llm=llm)
        payload: PipelineResponse = {
            "status": "smalltalk",
            "answer": reply,
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "branch_id": branch_id,
            "request_id": request_id,
            "timings": {"total_ms": _elapsed_ms(start_time)},
        }
        _log_request_success(
            config=config,
            request_id=request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            cache_hit=False,
            retrieval_ms=0.0,
            llm_ms=0.0,
            total_ms=(perf_counter() - start_time) * 1000.0,
            usage_stats={},
        )
        return payload

    scope = classify_scope(normalized_query, session_id=session_id, llm=llm)
    if not scope.allow:
        payload = {
            "status": "out_of_scope",
            "message": scope.user_message,
            "query": query,
            "scope_label": scope.label,
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "branch_id": branch_id,
            "request_id": request_id,
            "timings": {"total_ms": _elapsed_ms(start_time)},
        }
        _log_request_success(
            config=config,
            request_id=request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            cache_hit=False,
            retrieval_ms=0.0,
            llm_ms=0.0,
            total_ms=(perf_counter() - start_time) * 1000.0,
            usage_stats={},
        )
        return payload

    answer_cache_store = None
    answer_cache_fingerprint = ""
    answer_cache_lookup_ms = 0.0
    if _answer_cache_enabled(config):
        answer_cache_store = get_answer_cache_store(
            str(config.data_dir / "chroma"),
            embeddings_device=compute_device,
        )
        answer_cache_fingerprint = build_answer_cache_fingerprint(
            config=config,
            top_n=requested_top_n,
            include_paper_links=include_paper_links,
            backend="baseline",
        )
        answer_cache_lookup_start = perf_counter()
        cached_answer = lookup_answer_cache(
            normalized_query,
            store=answer_cache_store,
            config_fingerprint=answer_cache_fingerprint,
            ttl_seconds=int(getattr(config, "answer_cache_ttl_seconds", 604800)),
            min_similarity=float(getattr(config, "answer_cache_min_similarity", 0.9)),
            strict_fingerprint=bool(getattr(config, "answer_cache_strict_fingerprint", True)),
        )
        answer_cache_lookup_ms = (perf_counter() - answer_cache_lookup_start) * 1000.0
        if cached_answer is not None:
            payload = _build_answer_cache_hit_payload(
                cached_answer,
                request_id=request_id,
                query=query,
                branch_id=branch_id,
                lookup_ms=answer_cache_lookup_ms,
            )
            _pipeline_log(
                config.log_pipeline,
                "[PIPELINE] Answer cache hit | query_hash=%s similarity=%.3f match_type=%s",
                hash_query_text(query)[:12],
                float(payload.get("answer_cache_similarity", 0.0) or 0.0),
                str(payload.get("answer_cache_match_type", "") or "similar"),
            )
            _log_request_success(
                config=config,
                request_id=request_id,
                session_id=session_id,
                branch_id=branch_id,
                query=query,
                payload=payload,
                cache_hit=True,
                retrieval_ms=0.0,
                llm_ms=0.0,
                total_ms=(perf_counter() - start_time) * 1000.0,
                usage_stats={},
            )
            return payload

    context = _prepare_chat_context(
        query=normalized_query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=requested_top_n,
        llm=llm,
        scope=scope,
        config=config,
        request_id=request_id,
        compute_device=compute_device,
    )
    llm = context["llm"]
    context_top_k = int(context["context_top_k"])

    if llm is None:
        payload = {
            "status": "answered",
            "answer": "LLM not configured. Set NVIDIA_API_KEY to enable contextual answers.",
            "docs_preview": _filter_source_links(context["docs_preview"], include_paper_links),
            "pubmed_query": context["pubmed_query"],
            "scope_label": scope.label,
            "scope_message": scope.user_message,
            "reframed_query": scope.reframed_query or "",
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "cache_hit": context["cache_hit"],
            "branch_id": branch_id,
            "request_id": request_id,
            "timings": {
                "answer_cache_lookup_ms": round(answer_cache_lookup_ms, 3),
                "retrieval_ms": round(float(context["retrieval_ms"]), 3),
                "llm_ms": 0.0,
                "total_ms": _elapsed_ms(start_time),
            },
        }
        _log_request_success(
            config=config,
            request_id=request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            cache_hit=context["cache_hit"],
            retrieval_ms=float(context["retrieval_ms"]),
            llm_ms=0.0,
            total_ms=(perf_counter() - start_time) * 1000.0,
            usage_stats={},
        )
        return payload

    base_chain = build_rag_chain(
        llm,
        context["retriever"],
        max_abstracts=context_top_k,
        max_context_tokens=config.max_context_tokens,
        trim_strategy=config.context_trim_strategy,
    )
    chat_chain = build_chat_chain(base_chain)

    llm_start = perf_counter()
    with start_span("llm.generate", attributes={"request_id": request_id, "provider": "nvidia"}):
        raw_answer = chat_chain.invoke(
            {
                "input": query,
                "retrieval_query": context["retrieval_query"],
            },
            config={"configurable": {"session_id": session_id, "branch_id": branch_id}},
        )
    usage_stats = log_llm_usage("answer.invoke", raw_answer)
    llm_ms = _phase_elapsed_ms(llm_start, start_time)
    answer = _extract_message_text(raw_answer)

    retrieved_docs = _retrieve_docs_for_sources(
        retriever=context["retriever"],
        retrieval_query=context["retrieval_query"],
    )
    retrieved_contexts = _docs_to_eval_contexts(
        retrieved_docs,
        config=config,
        top_n=requested_top_n,
    )
    if config.citation_alignment:
        answer, alignment_issues = align_answer_citations(
            answer,
            contexts=retrieved_contexts,
            mode=config.alignment_mode,
        )
    else:
        alignment_issues = []
    sources = _expand_display_sources(
        _collect_sources_from_docs(
            retrieved_docs,
            top_n=requested_top_n,
            include_paper_links=include_paper_links,
        ),
        docs_preview=context["docs_preview"],
        top_n=requested_top_n,
        include_paper_links=include_paper_links,
    )
    answer, invalid_citations, evidence_quality = annotate_answer_metadata(
        answer,
        _source_pmids(sources),
    )
    validation_payload = _run_optional_validation(
        config=config,
        user_query=normalized_query,
        answer=answer,
        retrieved_docs=retrieved_docs,
        sources=sources,
        context_top_k=context_top_k,
        compute_device=compute_device,
    )
    _pipeline_log(
        config.log_pipeline,
        "[PIPELINE] Final source selection | requested_top_n=%s context_top_k=%s docs_preview=%s retrieved_docs=%s unique_retrieved_pmids=%s final_sources=%s",
        requested_top_n,
        context_top_k,
        len(context["docs_preview"]),
        len(retrieved_docs),
        _count_unique_pmids_from_docs(retrieved_docs),
        len(sources),
    )

    payload: PipelineResponse = {
        "status": "answered",
        "answer": answer,
        "sources": sources,
        "reranker_active": context["reranker_active"],
        "pubmed_query": context["pubmed_query"],
        "docs_preview": _filter_source_links(context["docs_preview"], include_paper_links),
        "scope_label": scope.label,
        "scope_message": scope.user_message,
        "reframed_query": scope.reframed_query or "",
        "reframe_note": context.get("reframe_note", ""),
        "query": query,
        "intent_label": intent_label,
        "intent_confidence": intent_confidence,
        "retrieved_contexts": retrieved_contexts,
        "cache_hit": context["cache_hit"],
        "cache_status": context["cache_status"],
        "branch_id": branch_id,
        "request_id": request_id,
        "timings": {
            "answer_cache_lookup_ms": round(answer_cache_lookup_ms, 3),
            "retrieval_ms": round(float(context["retrieval_ms"]), 3),
            "llm_ms": round(llm_ms, 3),
            "total_ms": _elapsed_ms(start_time),
        },
    }
    if alignment_issues:
        payload["alignment_issues"] = alignment_issues
    if invalid_citations:
        payload["invalid_citations"] = invalid_citations
    if evidence_quality:
        payload["evidence_quality"] = evidence_quality
    _add_source_count_note(
        payload,
        requested_top_n=requested_top_n,
        diagnostics=context.get("source_diagnostics"),
        log_pipeline=config.log_pipeline,
    )
    payload.update(validation_payload)
    if answer_cache_store is not None and answer_cache_fingerprint:
        store_answer_cache(
            normalized_query,
            response_payload=payload,
            config_fingerprint=answer_cache_fingerprint,
            store=answer_cache_store,
            model_id=str(getattr(config, "nvidia_model", "") or ""),
            backend="baseline",
        )
    _log_request_success(
        config=config,
        request_id=request_id,
        session_id=session_id,
        branch_id=branch_id,
        query=query,
        payload=payload,
        cache_hit=context["cache_hit"],
        retrieval_ms=float(context["retrieval_ms"]),
        llm_ms=llm_ms,
        total_ms=(perf_counter() - start_time) * 1000.0,
        usage_stats=usage_stats,
    )
    return payload


def _stream_chat_impl(
    *,
    query: str,
    session_id: str,
    branch_id: str = "main",
    top_n: int,
    config: AppConfig,
    request_id: str,
    include_paper_links: bool,
    start_time: float,
    compute_device: str | None = None,
):
    requested_top_n = _sanitize_top_n(top_n)
    normalized_query = normalize_user_query(query)
    llm = _get_llm_safe(config)
    intent = classify_intent_details(normalized_query, llm, log_enabled=config.log_pipeline)
    intent_label = str(intent.get("label", "")).lower()
    intent_confidence = float(intent.get("confidence", 0.0) or 0.0)
    if should_short_circuit_smalltalk(intent, normalized_query):
        reply = smalltalk_reply(query, llm=llm)
        yield reply
        payload: PipelineResponse = {
            "status": "smalltalk",
            "answer": reply,
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "branch_id": branch_id,
            "request_id": request_id,
            "timings": {"total_ms": _elapsed_ms(start_time)},
        }
        _log_request_success(
            config=config,
            request_id=request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            cache_hit=False,
            retrieval_ms=0.0,
            llm_ms=0.0,
            total_ms=(perf_counter() - start_time) * 1000.0,
            usage_stats={},
        )
        return payload

    scope = classify_scope(normalized_query, session_id=session_id, llm=llm)
    if not scope.allow:
        yield scope.user_message
        payload = {
            "status": "out_of_scope",
            "answer": scope.user_message,
            "scope_label": scope.label,
            "scope_message": scope.user_message,
            "reframed_query": scope.reframed_query or "",
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "branch_id": branch_id,
            "request_id": request_id,
            "timings": {"total_ms": _elapsed_ms(start_time)},
        }
        _log_request_success(
            config=config,
            request_id=request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            cache_hit=False,
            retrieval_ms=0.0,
            llm_ms=0.0,
            total_ms=(perf_counter() - start_time) * 1000.0,
            usage_stats={},
        )
        return payload

    answer_cache_store = None
    answer_cache_fingerprint = ""
    answer_cache_lookup_ms = 0.0
    if _answer_cache_enabled(config):
        answer_cache_store = get_answer_cache_store(
            str(config.data_dir / "chroma"),
            embeddings_device=compute_device,
        )
        answer_cache_fingerprint = build_answer_cache_fingerprint(
            config=config,
            top_n=requested_top_n,
            include_paper_links=include_paper_links,
            backend="baseline",
        )
        answer_cache_lookup_start = perf_counter()
        cached_answer = lookup_answer_cache(
            normalized_query,
            store=answer_cache_store,
            config_fingerprint=answer_cache_fingerprint,
            ttl_seconds=int(getattr(config, "answer_cache_ttl_seconds", 604800)),
            min_similarity=float(getattr(config, "answer_cache_min_similarity", 0.9)),
            strict_fingerprint=bool(getattr(config, "answer_cache_strict_fingerprint", True)),
        )
        answer_cache_lookup_ms = (perf_counter() - answer_cache_lookup_start) * 1000.0
        if cached_answer is not None:
            payload = _build_answer_cache_hit_payload(
                cached_answer,
                request_id=request_id,
                query=query,
                branch_id=branch_id,
                lookup_ms=answer_cache_lookup_ms,
            )
            cached_answer_text = str(payload.get("answer") or payload.get("message") or "")
            if cached_answer_text:
                yield cached_answer_text
            _pipeline_log(
                config.log_pipeline,
                "[PIPELINE] Answer cache hit | query_hash=%s similarity=%.3f match_type=%s",
                hash_query_text(query)[:12],
                float(payload.get("answer_cache_similarity", 0.0) or 0.0),
                str(payload.get("answer_cache_match_type", "") or "similar"),
            )
            _log_request_success(
                config=config,
                request_id=request_id,
                session_id=session_id,
                branch_id=branch_id,
                query=query,
                payload=payload,
                cache_hit=True,
                retrieval_ms=0.0,
                llm_ms=0.0,
                total_ms=(perf_counter() - start_time) * 1000.0,
                usage_stats={},
            )
            return payload

    context = _prepare_chat_context(
        query=normalized_query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=requested_top_n,
        llm=llm,
        scope=scope,
        config=config,
        request_id=request_id,
        compute_device=compute_device,
    )
    llm = context["llm"]
    context_top_k = int(context["context_top_k"])

    if llm is None:
        fallback = "LLM not configured. Set NVIDIA_API_KEY to enable contextual answers."
        yield fallback
        payload = {
            "status": "answered",
            "answer": fallback,
            "docs_preview": _filter_source_links(context["docs_preview"], include_paper_links),
            "pubmed_query": context["pubmed_query"],
            "scope_label": scope.label,
            "scope_message": scope.user_message,
            "reframed_query": scope.reframed_query or "",
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "cache_hit": context["cache_hit"],
            "branch_id": branch_id,
            "request_id": request_id,
            "timings": {
                "answer_cache_lookup_ms": round(answer_cache_lookup_ms, 3),
                "retrieval_ms": round(float(context["retrieval_ms"]), 3),
                "llm_ms": 0.0,
                "total_ms": _elapsed_ms(start_time),
            },
        }
        _log_request_success(
            config=config,
            request_id=request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            cache_hit=context["cache_hit"],
            retrieval_ms=float(context["retrieval_ms"]),
            llm_ms=0.0,
            total_ms=(perf_counter() - start_time) * 1000.0,
            usage_stats={},
        )
        return payload

    base_chain = build_rag_chain(
        llm,
        context["retriever"],
        max_abstracts=context_top_k,
        max_context_tokens=config.max_context_tokens,
        trim_strategy=config.context_trim_strategy,
    )
    chat_chain = build_chat_chain(base_chain)

    usage_candidate: Any | None = None
    answer_text = ""
    llm_start = perf_counter()
    with start_span("llm.generate", attributes={"request_id": request_id, "provider": "nvidia"}):
        for chunk in chat_chain.stream(
            {
                "input": query,
                "retrieval_query": context["retrieval_query"],
            },
            config={"configurable": {"session_id": session_id, "branch_id": branch_id}},
        ):
            if _has_usage_metadata(chunk):
                usage_candidate = chunk
            text = _coerce_stream_chunk(chunk)
            if not text:
                continue
            answer_text += text
            yield text
    usage_stats = log_llm_usage("answer.stream", usage_candidate)
    llm_ms = _phase_elapsed_ms(llm_start, start_time)

    retrieved_docs = _retrieve_docs_for_sources(
        retriever=context["retriever"],
        retrieval_query=context["retrieval_query"],
    )
    retrieved_contexts = _docs_to_eval_contexts(
        retrieved_docs,
        config=config,
        top_n=requested_top_n,
    )
    if config.citation_alignment:
        answer_text, alignment_issues = align_answer_citations(
            answer_text,
            contexts=retrieved_contexts,
            mode=config.alignment_mode,
        )
    else:
        alignment_issues = []
    sources = _expand_display_sources(
        _collect_sources_from_docs(
            retrieved_docs,
            top_n=requested_top_n,
            include_paper_links=include_paper_links,
        ),
        docs_preview=context["docs_preview"],
        top_n=requested_top_n,
        include_paper_links=include_paper_links,
    )
    answer_text, invalid_citations, evidence_quality = annotate_answer_metadata(
        answer_text,
        _source_pmids(sources),
    )
    validation_payload = _run_optional_validation(
        config=config,
        user_query=normalized_query,
        answer=answer_text,
        retrieved_docs=retrieved_docs,
        sources=sources,
        context_top_k=context_top_k,
        compute_device=compute_device,
    )
    _pipeline_log(
        config.log_pipeline,
        "[PIPELINE] Final source selection | requested_top_n=%s context_top_k=%s docs_preview=%s retrieved_docs=%s unique_retrieved_pmids=%s final_sources=%s",
        requested_top_n,
        context_top_k,
        len(context["docs_preview"]),
        len(retrieved_docs),
        _count_unique_pmids_from_docs(retrieved_docs),
        len(sources),
    )
    payload: PipelineResponse = {
        "status": "answered",
        "answer": answer_text,
        "sources": sources,
        "reranker_active": context["reranker_active"],
        "pubmed_query": context["pubmed_query"],
        "docs_preview": _filter_source_links(context["docs_preview"], include_paper_links),
        "scope_label": scope.label,
        "scope_message": scope.user_message,
        "reframed_query": scope.reframed_query or "",
        "reframe_note": context.get("reframe_note", ""),
        "query": query,
        "intent_label": intent_label,
        "intent_confidence": intent_confidence,
        "retrieved_contexts": retrieved_contexts,
        "cache_hit": context["cache_hit"],
        "cache_status": context["cache_status"],
        "branch_id": branch_id,
        "request_id": request_id,
        "timings": {
            "answer_cache_lookup_ms": round(answer_cache_lookup_ms, 3),
            "retrieval_ms": round(float(context["retrieval_ms"]), 3),
            "llm_ms": round(llm_ms, 3),
            "total_ms": _elapsed_ms(start_time),
        },
    }
    if alignment_issues:
        payload["alignment_issues"] = alignment_issues
    if invalid_citations:
        payload["invalid_citations"] = invalid_citations
    if evidence_quality:
        payload["evidence_quality"] = evidence_quality
    _add_source_count_note(
        payload,
        requested_top_n=requested_top_n,
        diagnostics=context.get("source_diagnostics"),
        log_pipeline=config.log_pipeline,
    )
    payload.update(validation_payload)
    if answer_cache_store is not None and answer_cache_fingerprint:
        store_answer_cache(
            normalized_query,
            response_payload=payload,
            config_fingerprint=answer_cache_fingerprint,
            store=answer_cache_store,
            model_id=str(getattr(config, "nvidia_model", "") or ""),
            backend="baseline",
        )
    _log_request_success(
        config=config,
        request_id=request_id,
        session_id=session_id,
        branch_id=branch_id,
        query=query,
        payload=payload,
        cache_hit=context["cache_hit"],
        retrieval_ms=float(context["retrieval_ms"]),
        llm_ms=llm_ms,
        total_ms=(perf_counter() - start_time) * 1000.0,
        usage_stats=usage_stats,
    )
    return payload


def _prepare_chat_context(
    *,
    query: str,
    session_id: str,
    branch_id: str = "main",
    top_n: int,
    llm: Any | None = None,
    scope: ScopeResult | None = None,
    config: AppConfig | None = None,
    request_id: str = "",
    executor_factory=ThreadPoolExecutor,
    compute_device: str | None = None,
) -> Dict[str, Any]:
    config = config or load_config()
    persist_dir = str(config.data_dir / "chroma")
    log_pipeline = bool(getattr(config, "log_pipeline", False))
    requested_top_n = _sanitize_top_n(top_n)
    context_top_k = _resolve_context_top_k(config, requested_top_n)
    retrieval_candidate_multiplier = max(
        1,
        int(getattr(config, "retrieval_candidate_multiplier", 3) or 3),
    )
    candidate_fetch_k = max(
        requested_top_n,
        min(50, requested_top_n * retrieval_candidate_multiplier),
    )
    cache_store_retmax = max(candidate_fetch_k, QUERY_CACHE_PMID_STORE_LIMIT)
    cache_ttl_seconds = int(getattr(config, "pubmed_cache_ttl_seconds", 604800))
    negative_cache_ttl_seconds = int(
        getattr(config, "pubmed_negative_cache_ttl_seconds", 3600)
    )
    llm = llm or _get_llm_safe(config)
    scope = scope or ScopeResult(
        label="BIOMEDICAL",
        allow=True,
        user_message="ok",
        reframed_query=None,
        reason="default",
    )

    start = perf_counter()
    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Query start | query_hash=%s requested_top_n=%s context_top_k=%s candidate_fetch_k=%s session_id=%s request_id=%s",
        hash_query_text(query)[:12],
        requested_top_n,
        context_top_k,
        candidate_fetch_k,
        session_id,
        request_id,
    )

    with executor_factory(max_workers=3) as executor:
        cache_future = executor.submit(
            get_query_cache_store,
            persist_dir,
            embeddings_device=compute_device,
        )
        abstract_future = executor.submit(
            get_abstract_store,
            persist_dir,
            embeddings_device=compute_device,
        )

        cache_store = cache_future.result()
        abstract_store = abstract_future.result()
        reranker_future = executor.submit(_prepare_reranker_resources, bool(config.use_reranker))

        pubmed_query = scope.reframed_query or query
        cache_payload = lookup_query_result_cache(
            query,
            store=cache_store,
            ttl_seconds=cache_ttl_seconds,
            negative_ttl_seconds=negative_cache_ttl_seconds,
        )
        use_cache = False
        if cache_payload and scope.reframed_query is None:
            use_cache = True
        elif cache_payload and scope.reframed_query:
            cached_query = str(cache_payload.get("pubmed_query", "") or "")
            if cached_query and cached_query == scope.reframed_query:
                use_cache = True

        cached_pmids = _normalize_pmid_list((cache_payload or {}).get("pmids"))
        if use_cache and len(cached_pmids) < candidate_fetch_k:
            use_cache = False
            _pipeline_log(
                log_pipeline,
                "[PIPELINE] Cache payload insufficient for requested_top_n | cached_pmids_len=%s requested_top_n=%s candidate_fetch_k=%s query_hash=%s",
                len(cached_pmids),
                requested_top_n,
                candidate_fetch_k,
                hash_query_text(query)[:12],
            )

        if use_cache:
            pubmed_query = str(cache_payload.get("pubmed_query") or pubmed_query)
            with start_span(
                "pubmed.retrieve",
                attributes={"request_id": request_id, "cache_hit": True},
            ):
                fetched_pmids, records, documents = _fetch_records_for_pmids(
                    cached_pmids,
                    target_count=candidate_fetch_k,
                )
            search_pmids = list(cached_pmids)
            _pipeline_log(
                log_pipeline,
                "[PIPELINE] Cache hit fetch | cached_pmids_len=%s sliced_pmids_len=%s records_len=%s documents_len=%s",
                len(cached_pmids),
                len(fetched_pmids),
                len(records),
                len(documents),
            )
            cache_status = "hit"
        else:
            if scope.reframed_query:
                pubmed_query = scope.reframed_query
            else:
                pubmed_query = rewrite_to_pubmed_query(query, llm)
            fetch_future: Future[tuple[list[str], list[str], list[dict[str, Any]], list[Any]]] = executor.submit(
                _fetch_pubmed_records,
                query,
                pubmed_query,
                candidate_fetch_k,
                requested_top_n,
                request_id,
                llm,
                bool(getattr(config, "multi_strategy_retrieval", True)),
            )
            search_pmids, fetched_pmids, records, documents = fetch_future.result()
            remember_query_result(
                query,
                pubmed_query=pubmed_query,
                pmids=search_pmids,
                requested_retmax=cache_store_retmax,
                store=cache_store,
            )
            _pipeline_log(
                log_pipeline,
                "[PIPELINE] PubMed fetch complete | esearch_pmids_len=%s sliced_pmids_len=%s records_len=%s documents_len=%s candidate_fetch_k=%s cache_store_retmax=%s",
                len(search_pmids),
                len(fetched_pmids),
                len(records),
                len(documents),
                candidate_fetch_k,
                cache_store_retmax,
            )
            cache_status = "miss"

        reranker_resources = reranker_future.result()

    source_diagnostics = _build_source_diagnostics(
        search_pmids=search_pmids,
        fetched_pmids=fetched_pmids,
        records=records,
        documents=documents,
    )

    if documents:
        embedded_count = upsert_abstracts(
            abstract_store,
            documents,
            query_text=query,
            pubmed_query=pubmed_query,
            log_pipeline=log_pipeline,
        )
    else:
        embedded_count = 0
        _pipeline_log(
            log_pipeline,
            "[PIPELINE] Embedding skipped | reason=no_documents query_hash=%s",
            hash_query_text(query)[:12],
        )
    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Abstract upsert status | embedded_count=%s documents_len=%s",
        embedded_count,
        len(documents),
    )

    if scope.reframed_query:
        retrieval_query = scope.reframed_query
    else:
        retrieval_query = build_contextual_retrieval_query(
            query,
            session_id,
            branch_id=branch_id,
            llm=llm,
            base_query=pubmed_query,
        )

    retriever, reranker_active = _build_retriever(
        abstract_store=abstract_store,
        top_n=requested_top_n,
        candidate_multiplier=retrieval_candidate_multiplier,
        use_reranker=bool(config.use_reranker),
        hybrid_retrieval=bool(config.hybrid_retrieval),
        hybrid_alpha=float(config.hybrid_alpha),
        log_pipeline=log_pipeline,
        reranker_resources=reranker_resources,
    )

    reframe_note = ""
    if scope.reframed_query and scope.user_message != "ok":
        reframe_note = scope.user_message

    retrieval_ms = (perf_counter() - start) * 1000.0
    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Retrieval config | requested_top_n=%s context_top_k=%s candidate_fetch_k=%s esearch_pmids=%s unique_record_pmids=%s documents_len=%s reranker_active=%s cache=%s abstracts_fetched=%s abstracts_embedded=%s retrieval_ms=%.2f",
        requested_top_n,
        context_top_k,
        candidate_fetch_k,
        source_diagnostics["esearch_pmids_len"],
        _count_unique_pmids_from_records(records),
        len(documents),
        reranker_active,
        cache_status,
        len(records),
        embedded_count,
        retrieval_ms,
    )

    return {
        "llm": llm,
        "retriever": retriever,
        "reranker_active": reranker_active,
        "pubmed_query": pubmed_query,
        "retrieval_query": retrieval_query,
        "reframe_note": reframe_note,
        "docs_preview": _build_docs_preview(records, top_n=requested_top_n),
        "cache_status": cache_status,
        "cache_hit": cache_status in {"hit", "hit_refreshed"},
        "context_top_k": context_top_k,
        "candidate_fetch_k": candidate_fetch_k,
        "abstracts_fetched": len(records),
        "abstracts_embedded": embedded_count,
        "source_diagnostics": source_diagnostics,
        "retrieval_ms": retrieval_ms,
    }


def _fetch_pubmed_records(
    user_query: str,
    pubmed_query: str,
    candidate_fetch_k: int,
    requested_top_n: int,
    request_id: str,
    llm: Any | None,
    multi_strategy_retrieval: bool,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[Any]]:
    with start_span(
        "pubmed.retrieve",
        attributes={"request_id": request_id, "pubmed_query_hash": hash_query_text(pubmed_query)},
    ):
        if multi_strategy_retrieval:
            queries = build_multi_strategy_queries(user_query or pubmed_query, llm)
            search_pmids = _normalize_pmid_list(
                multi_strategy_esearch(queries, retmax_each=max(requested_top_n, 12))
            )
        else:
            search_pmids = _normalize_pmid_list(
                multi_strategy_esearch(
                    [pubmed_query or user_query],
                    retmax_each=max(requested_top_n, 12),
                )
            )
        fetched_pmids, records, documents = _fetch_records_for_pmids(
            search_pmids,
            target_count=candidate_fetch_k,
        )
    return search_pmids, fetched_pmids, records, documents


def _fetch_records_for_pmids(
    pmids: list[str],
    *,
    target_count: int,
) -> tuple[list[str], list[dict[str, Any]], list[Any]]:
    normalized_pmids = _normalize_pmid_list(pmids)
    if not normalized_pmids:
        return [], [], []

    target_unique_count = max(1, int(target_count))
    records: list[dict[str, Any]] = []
    fetched_pmids: list[str] = []
    documents: list[Any] = []
    batch_size = target_unique_count
    offset = 0
    while offset < len(normalized_pmids):
        if (
            _count_unique_pmids_from_records(records) >= target_unique_count
            and _count_unique_pmids_from_docs(documents) >= target_unique_count
        ):
            break
        batch_pmids = normalized_pmids[offset : offset + batch_size]
        if not batch_pmids:
            break
        offset += len(batch_pmids)
        fetched_pmids.extend(batch_pmids)
        batch_records = pubmed_efetch(batch_pmids)
        if batch_records:
            records.extend(batch_records)
            documents = to_documents(records)
    if not documents:
        documents = to_documents(records)
    return fetched_pmids, records, documents


def _prepare_reranker_resources(use_reranker: bool):
    if not use_reranker:
        return None
    try:
        from langchain_community.document_compressors import FlashrankRerank
    except Exception:
        return None
    try:
        return FlashrankRerank()
    except Exception:
        return None


def _build_docs_preview(records: List[Dict[str, Any]], top_n: int) -> List[SourceItem]:
    items: list[SourceItem] = []
    seen: set[str] = set()
    for record in records:
        pmid = str(record.get("pmid", "")).strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        item: SourceItem = {
            "rank": len(items) + 1,
            "pmid": pmid,
            "title": str(record.get("title", "") or ""),
            "year": str(record.get("year", "") or ""),
            "journal": str(record.get("journal", "") or ""),
            "doi": str(record.get("doi", "") or ""),
            "fulltext_url": str(record.get("fulltext_url", "") or ""),
        }
        pmcid = str(record.get("pmcid", "") or "").strip()
        if pmcid:
            item["pmcid"] = pmcid
        items.append(item)
        if len(items) >= top_n:
            break
    return items


def _retrieve_docs_for_sources(retriever, retrieval_query: str) -> list:
    try:
        docs = retriever.invoke(retrieval_query)
    except Exception:
        docs = []
    if not isinstance(docs, list):
        return list(docs or [])
    return docs


def _count_unique_pmids_from_records(records: List[Dict[str, Any]]) -> int:
    seen: set[str] = set()
    for record in records or []:
        pmid = str((record or {}).get("pmid", "") or "").strip()
        if pmid:
            seen.add(pmid)
    return len(seen)


def _count_unique_pmids_from_docs(docs: list) -> int:
    seen: set[str] = set()
    for doc in docs or []:
        metadata = getattr(doc, "metadata", {}) or {}
        pmid = str(metadata.get("pmid", "") or "").strip()
        if pmid:
            seen.add(pmid)
    return len(seen)


def _normalize_pmid_list(pmids: Any) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_pmid in pmids or []:
        pmid = str(raw_pmid or "").strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        normalized.append(pmid)
    return normalized


def _record_pmid_diagnostics(records: list[dict[str, Any]]) -> dict[str, int]:
    seen: set[str] = set()
    missing = 0
    duplicates = 0
    for record in records or []:
        pmid = str((record or {}).get("pmid", "") or "").strip()
        if not pmid:
            missing += 1
            continue
        if pmid in seen:
            duplicates += 1
            continue
        seen.add(pmid)
    return {
        "records_len": len(records or []),
        "unique_record_pmids": len(seen),
        "missing_record_pmids": missing,
        "duplicate_record_pmids": duplicates,
    }


def _document_pmid_diagnostics(documents: list[Any]) -> dict[str, int]:
    seen: set[str] = set()
    missing = 0
    duplicates = 0
    for doc in documents or []:
        metadata = getattr(doc, "metadata", {}) or {}
        pmid = str(metadata.get("pmid", "") or "").strip()
        if not pmid:
            missing += 1
            continue
        if pmid in seen:
            duplicates += 1
            continue
        seen.add(pmid)
    return {
        "documents_len": len(documents or []),
        "unique_document_pmids": len(seen),
        "missing_document_pmids": missing,
        "duplicate_document_pmids": duplicates,
    }


def _build_source_diagnostics(
    *,
    search_pmids: list[str],
    fetched_pmids: list[str],
    records: list[dict[str, Any]],
    documents: list[Any],
) -> dict[str, int]:
    diagnostics = {
        "esearch_pmids_len": len(_normalize_pmid_list(search_pmids)),
        "fetched_pmids_len": len(_normalize_pmid_list(fetched_pmids)),
    }
    diagnostics.update(_record_pmid_diagnostics(records))
    diagnostics.update(_document_pmid_diagnostics(documents))
    return diagnostics


def _collect_sources_from_docs(
    docs: list,
    *,
    top_n: int,
    include_paper_links: bool,
) -> List[SourceItem]:
    sources: list[SourceItem] = []
    seen_pmids: set[str] = set()
    for doc in select_context_documents(docs, max_abstracts=top_n):
        meta = getattr(doc, "metadata", {}) or {}
        pmid = str(meta.get("pmid", "")).strip()
        if not pmid or pmid in seen_pmids:
            continue
        item: SourceItem = {
            "rank": len(sources) + 1,
            "pmid": pmid,
            "title": str(meta.get("title", "") or ""),
            "journal": str(meta.get("journal", "") or ""),
            "year": str(meta.get("year", "") or ""),
        }
        if include_paper_links:
            item["doi"] = str(meta.get("doi", "") or "")
            item["fulltext_url"] = str(meta.get("fulltext_url", "") or "")
            pmcid = str(meta.get("pmcid", "") or "").strip()
            if pmcid:
                item["pmcid"] = pmcid
        context = _extract_doc_context(doc)
        if context:
            item["context"] = context
        sources.append(item)
        seen_pmids.add(pmid)
        if len(sources) >= top_n:
            break
    return sources


def _expand_display_sources(
    sources: list[SourceItem],
    *,
    docs_preview: list[SourceItem],
    top_n: int,
    include_paper_links: bool,
) -> list[SourceItem]:
    expanded = [dict(item) for item in (sources or []) if isinstance(item, Mapping)]
    seen_pmids = {
        str(item.get("pmid", "") or "").strip()
        for item in expanded
        if str(item.get("pmid", "") or "").strip()
    }
    preview_items = _filter_source_links(docs_preview or [], include_paper_links)
    for preview_item in preview_items:
        pmid = str(preview_item.get("pmid", "") or "").strip()
        if not pmid or pmid in seen_pmids:
            continue
        item: SourceItem = {
            "rank": len(expanded) + 1,
            "pmid": pmid,
            "title": str(preview_item.get("title", "") or ""),
            "journal": str(preview_item.get("journal", "") or ""),
            "year": str(preview_item.get("year", "") or ""),
        }
        if include_paper_links:
            item["doi"] = str(preview_item.get("doi", "") or "")
            item["fulltext_url"] = str(preview_item.get("fulltext_url", "") or "")
            pmcid = str(preview_item.get("pmcid", "") or "").strip()
            if pmcid:
                item["pmcid"] = pmcid
        expanded.append(item)
        seen_pmids.add(pmid)
        if len(expanded) >= max(1, int(top_n)):
            break
    for index, item in enumerate(expanded, start=1):
        item["rank"] = index
    return expanded[: max(1, int(top_n))]


def _extract_doc_context(doc: Any, limit: int = 1800) -> str:
    text = str(getattr(doc, "page_content", "") or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(parts) >= 2:
        text = "\n\n".join(parts[1:])
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _build_answer_cache_hit_payload(
    cached_answer: Mapping[str, Any],
    *,
    request_id: str,
    query: str,
    branch_id: str,
    lookup_ms: float,
) -> PipelineResponse:
    response_payload = dict(cached_answer.get("response_payload", {}) or {})
    payload: PipelineResponse = dict(response_payload)
    payload["query"] = query
    payload["branch_id"] = branch_id
    payload["request_id"] = request_id
    payload["answer_cache_hit"] = True
    payload["answer_cache_match_type"] = str(cached_answer.get("match_type", "similar") or "similar")
    payload["answer_cache_created_at"] = str(cached_answer.get("created_at", "") or "")
    payload["answer_cache_similarity"] = float(cached_answer.get("similarity", 0.0) or 0.0)
    payload["answer_cache_query"] = str(cached_answer.get("matched_query", "") or "")
    payload["answer_cache_config_match"] = bool(cached_answer.get("config_match", False))
    payload["cache_hit"] = True
    payload["cache_status"] = "answer_hit"
    note = str(cached_answer.get("note", "") or "").strip()
    if note:
        payload["answer_cache_note"] = note
    existing_timings = payload.get("timings")
    timing_payload = dict(existing_timings or {}) if isinstance(existing_timings, Mapping) else {}
    timing_payload["answer_cache_lookup_ms"] = round(lookup_ms, 3)
    timing_payload["total_ms"] = round(lookup_ms, 3)
    payload["timings"] = timing_payload
    _apply_answer_metadata_to_payload(payload)
    return payload


def _apply_answer_metadata_to_payload(payload: dict[str, Any]) -> None:
    answer_text = str(payload.get("answer") or payload.get("message") or "").strip()
    if not answer_text:
        return
    cleaned_answer, invalid_citations, evidence_quality = annotate_answer_metadata(
        answer_text,
        _source_pmids(payload.get("sources", []) or []),
    )
    if "answer" in payload:
        payload["answer"] = cleaned_answer
    elif "message" in payload:
        payload["message"] = cleaned_answer
    if invalid_citations:
        payload["invalid_citations"] = invalid_citations
    elif "invalid_citations" in payload:
        payload.pop("invalid_citations", None)
    if evidence_quality:
        payload["evidence_quality"] = evidence_quality
    elif "evidence_quality" in payload:
        payload.pop("evidence_quality", None)


def _source_pmids(sources: list[Mapping[str, Any]] | list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("pmid", "") or "").strip()
        for item in sources or []
        if str(item.get("pmid", "") or "").strip()
    ]


def _add_source_count_note(
    payload: dict[str, Any],
    *,
    requested_top_n: int,
    diagnostics: Mapping[str, Any] | None = None,
    log_pipeline: bool = False,
) -> None:
    sources = payload.get("sources", []) or []
    if not isinstance(sources, list):
        return
    if len(sources) >= max(1, int(requested_top_n)):
        payload.pop("source_count_note", None)
        return

    requested_count = max(1, int(requested_top_n))
    note = f"Only {len(sources)} unique papers were available for this query."
    diagnostics_map = dict(diagnostics or {}) if isinstance(diagnostics, Mapping) else {}
    if diagnostics_map:
        if int(diagnostics_map.get("esearch_pmids_len", 0) or 0) < requested_count:
            note = f"PubMed returned fewer than {requested_count} records."
        elif int(diagnostics_map.get("records_len", 0) or 0) < requested_count:
            note = f"PubMed returned fewer than {requested_count} records."
        elif int(diagnostics_map.get("missing_record_pmids", 0) or 0) > 0 or int(
            diagnostics_map.get("missing_document_pmids", 0) or 0
        ) > 0:
            note = "Some records missing PMID metadata."
        elif int(diagnostics_map.get("duplicate_record_pmids", 0) or 0) > 0 or int(
            diagnostics_map.get("duplicate_document_pmids", 0) or 0
        ) > 0:
            note = "Duplicate PMIDs removed."

    payload["source_count_note"] = note
    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Source shortfall | requested_top_n=%s sources_len=%s note=%s esearch_pmids_len=%s fetched_pmids_len=%s records_len=%s unique_record_pmids=%s missing_record_pmids=%s duplicate_record_pmids=%s documents_len=%s unique_document_pmids=%s missing_document_pmids=%s duplicate_document_pmids=%s",
        requested_count,
        len(sources),
        note,
        diagnostics_map.get("esearch_pmids_len", 0),
        diagnostics_map.get("fetched_pmids_len", 0),
        diagnostics_map.get("records_len", 0),
        diagnostics_map.get("unique_record_pmids", 0),
        diagnostics_map.get("missing_record_pmids", 0),
        diagnostics_map.get("duplicate_record_pmids", 0),
        diagnostics_map.get("documents_len", 0),
        diagnostics_map.get("unique_document_pmids", 0),
        diagnostics_map.get("missing_document_pmids", 0),
        diagnostics_map.get("duplicate_document_pmids", 0),
    )


def _answer_cache_enabled(config: Any) -> bool:
    return bool(
        hasattr(config, "data_dir")
        and hasattr(config, "answer_cache_ttl_seconds")
        and hasattr(config, "answer_cache_min_similarity")
    )


def _docs_to_eval_contexts(
    docs: list,
    *,
    config: AppConfig,
    top_n: int,
) -> list[dict[str, str]]:
    return build_context_rows(
        select_context_documents(docs, max_abstracts=top_n),
        max_abstracts=top_n,
        max_context_tokens=config.max_context_tokens,
        trim_strategy=config.context_trim_strategy,
    )


def _build_retriever(
    *,
    abstract_store: Any,
    top_n: int,
    candidate_multiplier: int,
    use_reranker: bool,
    hybrid_retrieval: bool,
    hybrid_alpha: float,
    log_pipeline: bool = False,
    reranker_resources: Any | None = None,
):
    candidate_k = min(50, max(top_n, top_n * max(1, int(candidate_multiplier or 1))))
    retriever = _PipelineRetriever(
        abstract_store=abstract_store,
        candidate_k=candidate_k,
        final_k=top_n,
        compressor=reranker_resources if use_reranker else None,
        hybrid_retrieval=hybrid_retrieval,
        hybrid_alpha=hybrid_alpha,
        log_pipeline=log_pipeline,
    )
    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Built retriever | candidate_k=%s final_k=%s hybrid=%s reranker=%s",
        candidate_k,
        top_n,
        hybrid_retrieval,
        reranker_resources is not None,
    )
    return retriever, bool(reranker_resources is not None and use_reranker)


def _sanitize_top_n(top_n: int) -> int:
    try:
        value = int(top_n)
    except (TypeError, ValueError):
        return 10
    return max(1, min(10, value))


def _resolve_context_top_k(config: Any, requested_top_n: int) -> int:
    raw_value = getattr(config, "max_context_abstracts", getattr(config, "max_abstracts", requested_top_n))
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = requested_top_n
    return max(1, min(int(requested_top_n), parsed))


def _get_llm_safe(config: AppConfig | None = None):
    try:
        if config is not None:
            return get_nvidia_llm(model_name=config.nvidia_model, api_key=config.nvidia_api_key)
        return get_nvidia_llm()
    except Exception:
        return None


def _extract_message_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(message)


def _coerce_stream_chunk(chunk: Any) -> str:
    return _extract_message_text(chunk)


def _has_usage_metadata(response: Any) -> bool:
    if response is None:
        return False
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        return True
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping) and (
        response_metadata.get("token_usage") or response_metadata.get("usage")
    ):
        return True
    return False


def _is_followup_query(normalized_query: str) -> bool:
    tokens = normalized_query.split()
    if len(tokens) < 8:
        return True
    if any(phrase in normalized_query for phrase in FOLLOWUP_PATTERNS):
        return True
    if PRONOUN_PATTERN.search(normalized_query):
        return True
    return False


def _get_history_excerpt(
    session_id: str,
    *,
    branch_id: str = "main",
    max_messages: int = 6,
    max_chars: int = 1000,
) -> str:
    try:
        history = get_session_history(session_id, branch_id)
    except Exception:
        return ""

    messages = getattr(history, "messages", [])[-max_messages:]
    lines = []
    for message in messages:
        role = getattr(message, "type", "") or ""
        content = getattr(message, "content", "") or ""
        if not content:
            continue
        label = "assistant" if role in {"ai", "assistant"} else "user"
        lines.append(f"{label}: {content}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _get_last_assistant_excerpt(
    session_id: str,
    *,
    branch_id: str = "main",
    max_chars: int = 240,
) -> str:
    try:
        history = get_session_history(session_id, branch_id)
    except Exception:
        return ""

    for message in reversed(getattr(history, "messages", [])):
        role = getattr(message, "type", "") or ""
        if role not in {"ai", "assistant"}:
            continue
        content = getattr(message, "content", "") or ""
        if not content:
            continue
        cleaned = " ".join(content.strip().split())
        return cleaned[:max_chars]
    return ""


def _invoke_llm(llm: Any, prompt: str, usage_tag: str | None = None) -> str | None:
    try:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
            if usage_tag:
                log_llm_usage(usage_tag, result)
            return _extract_message_text(result)
        if hasattr(llm, "predict"):
            result = llm.predict(prompt)
            return _extract_message_text(result)
    except Exception:
        return None
    return None


def _run_optional_validation(
    *,
    config: AppConfig,
    user_query: str,
    answer: str,
    retrieved_docs: list,
    sources: list[SourceItem],
    context_top_k: int,
    compute_device: str | None = None,
) -> Dict[str, Any]:
    if not getattr(config, "validator_enabled", False):
        return {}
    if not answer.strip():
        return {}

    context = (
        _format_docs(
            retrieved_docs,
            max_abstracts=context_top_k,
            max_context_tokens=config.max_context_tokens,
            trim_strategy=config.context_trim_strategy,
        )
        if retrieved_docs
        else ""
    )
    source_pmids = [str(item.get("pmid", "")).strip() for item in sources if item.get("pmid")]
    result = validate_answer(
        user_query=user_query,
        answer=answer,
        context=context,
        source_pmids=source_pmids,
        model_name=str(
            getattr(config, "validator_model_name", "MoritzLaurer/DeBERTa-v3-base-mnli")
        ),
        threshold=float(getattr(config, "validator_threshold", 0.7)),
        margin=float(getattr(config, "validator_margin", 0.2)),
        max_premise_tokens=int(getattr(config, "validator_max_premise_tokens", 384)),
        max_hypothesis_tokens=int(getattr(config, "validator_max_hypothesis_tokens", 128)),
        max_length=int(getattr(config, "validator_max_length", 512)),
        top_n_chunks=int(getattr(config, "validator_top_n_chunks", 4)),
        top_k_sentences=int(getattr(config, "validator_top_k_sentences", 2)),
        retrieved_docs=retrieved_docs,
        device=compute_device,
    )
    _pipeline_log(
        getattr(config, "log_pipeline", False),
        "[PIPELINE] Validator result | valid=%s label=%s score=%.4f",
        result.get("valid", True),
        result.get("label", "UNKNOWN"),
        float(result.get("score", 0.0) or 0.0),
    )
    if result.get("valid", True):
        return {}

    details = result.get("details") or {}
    detail_reason = str(details.get("reason", "")).strip()
    detail_issues = details.get("issues") or []
    if not isinstance(detail_issues, list):
        detail_issues = [str(detail_issues)]

    issues = [str(item).strip() for item in detail_issues if str(item).strip()]
    if detail_reason and detail_reason not in issues:
        issues.append(detail_reason)
    if not issues:
        issues = ["Validator flagged low support between evidence and answer."]

    warning = "Some parts of the answer may not be fully supported by the retrieved abstracts."
    payload: Dict[str, Any] = {
        "validation_warning": warning,
        "validation_issues": issues,
        "validation_confidence": f"{float(result.get('score', 0.0) or 0.0):.3f}",
    }
    suggested_fix = details.get("suggested_fix")
    if suggested_fix:
        payload["validation_suggested_fix"] = str(suggested_fix)
    return payload


def _sanitize_query(text: str, fallback: str) -> str:
    cleaned = str(text).strip().strip('"').strip("'")
    if not cleaned:
        return fallback
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    if len(cleaned) > 300:
        cleaned = cleaned[:300].rstrip()
    return cleaned or fallback


def _filter_source_links(items: list[dict[str, Any]], include_paper_links: bool) -> list[dict[str, Any]]:
    if include_paper_links:
        return [dict(item) for item in items]
    filtered: list[dict[str, Any]] = []
    for item in items:
        copy = dict(item)
        copy.pop("doi", None)
        copy.pop("pmcid", None)
        copy.pop("fulltext_url", None)
        filtered.append(copy)
    return filtered


def _log_request_success(
    *,
    config: AppConfig,
    request_id: str,
    session_id: str,
    branch_id: str,
    query: str,
    payload: Mapping[str, Any],
    cache_hit: bool,
    retrieval_ms: float,
    llm_ms: float,
    total_ms: float,
    usage_stats: Mapping[str, Any],
) -> None:
    log_event(
        "request.complete",
        request_id=request_id,
        session_id=session_id,
        branch_id=branch_id,
        query_hash=hash_query_text(query),
        status=str(payload.get("status", "")),
        provider="nvidia" if config.nvidia_api_key else None,
        cache_hit=cache_hit,
        pmid_count=len(payload.get("sources", []) or []),
        retrieval_ms=round(retrieval_ms, 3),
        llm_ms=round(llm_ms, 3),
        total_ms=round(total_ms, 3),
        prompt_tokens=usage_stats.get("prompt_tokens"),
        completion_tokens=usage_stats.get("completion_tokens"),
        total_tokens=usage_stats.get("total_tokens"),
        store_path=str(config.metrics_store_path) if config.metrics_mode else None,
    )


def _log_request_error(
    *,
    config: AppConfig,
    request_id: str,
    session_id: str,
    branch_id: str,
    query: str,
    started_at: float,
    error: Exception,
) -> None:
    log_event(
        "request.error",
        request_id=request_id,
        session_id=session_id,
        branch_id=branch_id,
        query_hash=hash_query_text(query),
        error_type=error.__class__.__name__,
        error_message=str(error),
        total_ms=round((perf_counter() - started_at) * 1000.0, 3),
        provider="nvidia" if config.nvidia_api_key else None,
        store_path=str(config.metrics_store_path) if config.metrics_mode else None,
    )


def _pipeline_log(enabled: bool, message: str, *args: Any) -> None:
    if not enabled:
        return
    LOGGER.info(message, *args)


def _trim_text(text: str, limit: int = 140) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _elapsed_ms(started_at: float) -> float:
    if started_at <= 0:
        return 0.0
    return round((perf_counter() - started_at) * 1000.0, 3)


def _phase_elapsed_ms(phase_started_at: float, request_started_at: float) -> float:
    if request_started_at <= 0:
        return 0.0
    return round((perf_counter() - phase_started_at) * 1000.0, 3)
