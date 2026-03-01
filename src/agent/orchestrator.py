from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from typing import Any, Callable, Mapping
import logging

from src.agent.state import AgentState
from src.agent.tools import (
    answer_synthesis_stream_tool,
    answer_synthesis_tool,
    citation_formatting_tool,
    context_export_tool,
    pubmed_search_tool,
    query_refinement_tool,
    retriever_tool,
    safety_guardrail_tool,
)
from src.core.config import load_config
from src.integrations.nvidia import get_nvidia_llm
from src.integrations.storage import (
    build_answer_cache_fingerprint,
    get_answer_cache_store,
    lookup_answer_cache,
    store_answer_cache,
)
from src.logging_utils import hash_query_text, log_event
from src.types import PipelineResponse
from src.utils.answers import annotate_answer_metadata
from src.validators import validate_answer

LOGGER = logging.getLogger("agent.orchestrator")


def invoke_agent_chat(
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
    current_request_id = request_id or ""
    start_time = perf_counter()
    llm = _get_llm_safe()
    safe_top_n = _sanitize_top_n(top_n)
    answer_cache_store = get_answer_cache_store(
        str(config.data_dir / "chroma"),
        embeddings_device=compute_device,
    )
    answer_cache_fingerprint = build_answer_cache_fingerprint(
        config=config,
        top_n=safe_top_n,
        include_paper_links=include_paper_links,
        backend="agent",
    )
    answer_cache_lookup_start = perf_counter()
    cached_answer = lookup_answer_cache(
        query,
        store=answer_cache_store,
        config_fingerprint=answer_cache_fingerprint,
        ttl_seconds=int(getattr(config, "answer_cache_ttl_seconds", 604800)),
        min_similarity=float(getattr(config, "answer_cache_min_similarity", 0.9)),
        strict_fingerprint=bool(getattr(config, "answer_cache_strict_fingerprint", True)),
    )
    answer_cache_lookup_ms = (perf_counter() - answer_cache_lookup_start) * 1000.0
    if cached_answer is not None:
        payload = _build_cached_answer_payload(
            cached_answer,
            request_id=current_request_id,
            query=query,
            branch_id=branch_id,
            lookup_ms=answer_cache_lookup_ms,
        )
        _log_agent_request(
            config=config,
            request_id=current_request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            started_at=start_time,
        )
        return payload

    initial: AgentState = {
        "query": query,
        "session_id": session_id,
        "branch_id": branch_id,
        "top_n": safe_top_n,
        "llm": llm,
        "persist_dir": str(config.data_dir / "chroma"),
        "use_reranker": bool(config.use_reranker),
        "log_pipeline": bool(config.log_pipeline),
        "compute_device": str(compute_device or "cpu"),
    }

    state = _run_agent(initial, use_langgraph=bool(config.agent_use_langgraph))
    payload = _state_to_payload(
        state,
        request_id=current_request_id,
        include_paper_links=include_paper_links,
    )
    _apply_answer_metadata_to_payload(payload)
    payload.update(
        _run_optional_validation(
            config=config,
            query=query,
            answer=str(payload.get("answer", "") or ""),
            contexts=state.get("retrieved_contexts", []) or [],
            sources=payload.get("sources", []) or [],
            compute_device=compute_device,
        )
    )
    timings = dict(payload.get("timings", {}) or {})
    timings.setdefault("answer_cache_lookup_ms", round(answer_cache_lookup_ms, 3))
    timings["total_ms"] = round((perf_counter() - start_time) * 1000.0, 3)
    payload["timings"] = timings
    payload["branch_id"] = branch_id
    store_answer_cache(
        query,
        response_payload=payload,
        config_fingerprint=answer_cache_fingerprint,
        store=answer_cache_store,
        model_id=str(getattr(config, "nvidia_model", "") or ""),
        backend="agent",
    )
    _log_agent_request(
        config=config,
        request_id=current_request_id,
        session_id=session_id,
        branch_id=branch_id,
        query=query,
        payload=payload,
        started_at=start_time,
    )
    return payload


def stream_agent_chat(
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
    current_request_id = request_id or ""
    start_time = perf_counter()
    llm = _get_llm_safe()
    safe_top_n = _sanitize_top_n(top_n)
    answer_cache_store = get_answer_cache_store(
        str(config.data_dir / "chroma"),
        embeddings_device=compute_device,
    )
    answer_cache_fingerprint = build_answer_cache_fingerprint(
        config=config,
        top_n=safe_top_n,
        include_paper_links=include_paper_links,
        backend="agent",
    )
    answer_cache_lookup_start = perf_counter()
    cached_answer = lookup_answer_cache(
        query,
        store=answer_cache_store,
        config_fingerprint=answer_cache_fingerprint,
        ttl_seconds=int(getattr(config, "answer_cache_ttl_seconds", 604800)),
        min_similarity=float(getattr(config, "answer_cache_min_similarity", 0.9)),
        strict_fingerprint=bool(getattr(config, "answer_cache_strict_fingerprint", True)),
    )
    answer_cache_lookup_ms = (perf_counter() - answer_cache_lookup_start) * 1000.0
    if cached_answer is not None:
        payload = _build_cached_answer_payload(
            cached_answer,
            request_id=current_request_id,
            query=query,
            branch_id=branch_id,
            lookup_ms=answer_cache_lookup_ms,
        )
        cached_text = str(payload.get("answer") or payload.get("message") or "")
        if cached_text:
            yield cached_text
        _log_agent_request(
            config=config,
            request_id=current_request_id,
            session_id=session_id,
            branch_id=branch_id,
            query=query,
            payload=payload,
            started_at=start_time,
        )
        return payload

    guard = safety_guardrail_tool(
        query,
        session_id=session_id,
        llm=llm,
        log_pipeline=bool(config.log_pipeline),
    )
    scope = guard.get("scope")
    status = str(guard.get("status", "out_of_scope"))
    intent_label = str(guard.get("intent_label", "medical"))
    intent_confidence = float(guard.get("intent_confidence", 0.0) or 0.0)
    if not bool(guard.get("allow")):
        message = str(guard.get("message", "") or "")
        yield message
        return {
            "status": status,
            "answer": message,
            "query": query,
            "sources": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "scope_label": getattr(scope, "label", ""),
            "scope_message": getattr(scope, "user_message", message),
            "reframed_query": getattr(scope, "reframed_query", "") or "",
            "retrieved_contexts": [],
            "branch_id": branch_id,
            "request_id": current_request_id,
            "timings": {"total_ms": round((perf_counter() - start_time) * 1000.0, 3)},
        }

    refinement = query_refinement_tool(
        query,
        scope=scope,
        llm=llm,
    )
    pubmed_query = refinement["pubmed_query"]
    retrieval_query = refinement["retrieval_query"]
    reframe_note = refinement["reframe_note"]
    search = pubmed_search_tool(
        query,
        pubmed_query=pubmed_query,
        top_n=safe_top_n,
        persist_dir=str(config.data_dir / "chroma"),
        log_pipeline=bool(config.log_pipeline),
        compute_device=compute_device,
    )
    retrieve = retriever_tool(
        abstract_store=search["abstract_store"],
        retrieval_query=retrieval_query,
        top_n=safe_top_n,
        use_reranker=bool(config.use_reranker),
        log_pipeline=bool(config.log_pipeline),
    )
    contexts = context_export_tool(retrieve["docs"], top_n=safe_top_n)
    sources = citation_formatting_tool(retrieve["docs"], top_n=safe_top_n)
    sources = _filter_source_links(sources, include_paper_links=include_paper_links)
    docs_preview = search["docs_preview"]

    stream = answer_synthesis_stream_tool(
        query=query,
        retrieval_query=retrieval_query,
        session_id=session_id,
        branch_id=branch_id,
        llm=llm,
        retriever=retrieve["retriever"],
        context_top_k=int(search.get("context_top_k", safe_top_n) or safe_top_n),
    )
    answer_text = ""
    while True:
        try:
            chunk = next(stream)
        except StopIteration as stop:
            final = stop.value or {}
            answer_text = str(final.get("answer") or answer_text)
            break
        answer_text += str(chunk)
        yield str(chunk)

    answer_text, invalid_citations, evidence_quality = annotate_answer_metadata(
        answer_text,
        _source_pmids(sources),
    )

    payload: PipelineResponse = {
        "status": "answered",
        "answer": answer_text,
        "query": query,
        "sources": sources,
        "docs_preview": docs_preview,
        "pubmed_query": pubmed_query,
        "reranker_active": bool(retrieve.get("reranker_active", False)),
        "scope_label": getattr(scope, "label", "BIOMEDICAL"),
        "scope_message": getattr(scope, "user_message", "ok"),
        "reframed_query": getattr(scope, "reframed_query", "") or "",
        "reframe_note": reframe_note,
        "intent_label": intent_label,
        "intent_confidence": intent_confidence,
        "retrieved_contexts": contexts,
        "branch_id": branch_id,
        "request_id": current_request_id,
        "timings": {
            "answer_cache_lookup_ms": round(answer_cache_lookup_ms, 3),
            "total_ms": round((perf_counter() - start_time) * 1000.0, 3),
        },
    }
    if invalid_citations:
        payload["invalid_citations"] = invalid_citations
    if evidence_quality:
        payload["evidence_quality"] = evidence_quality
    payload.update(
        _run_optional_validation(
            config=config,
            query=query,
            answer=answer_text,
            contexts=contexts,
            sources=sources,
            compute_device=compute_device,
        )
    )
    _add_source_count_note(payload, requested_top_n=safe_top_n)
    store_answer_cache(
        query,
        response_payload=payload,
        config_fingerprint=answer_cache_fingerprint,
        store=answer_cache_store,
        model_id=str(getattr(config, "nvidia_model", "") or ""),
        backend="agent",
    )
    _log_agent_request(
        config=config,
        request_id=current_request_id,
        session_id=session_id,
        branch_id=branch_id,
        query=query,
        payload=payload,
        started_at=start_time,
    )
    return payload


def _run_agent(initial: AgentState, *, use_langgraph: bool) -> AgentState:
    if use_langgraph:
        runner = _get_langgraph_runner()
        if runner is not None:
            try:
                return runner(initial)
            except Exception as exc:
                LOGGER.warning("LangGraph runner failed, falling back to sequential path (%s)", exc)
    return _run_sequential(initial)


def _run_sequential(state: AgentState) -> AgentState:
    guard = safety_guardrail_tool(
        state["query"],
        session_id=state["session_id"],
        llm=state.get("llm"),
        log_pipeline=bool(state.get("log_pipeline", False)),
    )
    state.update(
        {
            "status": str(guard.get("status", "out_of_scope")),
            "intent_label": str(guard.get("intent_label", "medical")),
            "intent_confidence": float(guard.get("intent_confidence", 0.0) or 0.0),
        }
    )
    scope = guard.get("scope")
    if not bool(guard.get("allow")):
        state["answer"] = str(guard.get("message", "") or "")
        state["scope_label"] = getattr(scope, "label", "")
        state["scope_message"] = getattr(scope, "user_message", state["answer"])
        state["reframed_query"] = getattr(scope, "reframed_query", "") or ""
        return state

    refinement = query_refinement_tool(
        state["query"],
        scope=scope,
        llm=state.get("llm"),
    )
    state["pubmed_query"] = refinement["pubmed_query"]
    state["retrieval_query"] = refinement["retrieval_query"]
    state["reframe_note"] = refinement["reframe_note"]
    state["scope_label"] = getattr(scope, "label", "BIOMEDICAL")
    state["scope_message"] = getattr(scope, "user_message", "ok")
    state["reframed_query"] = getattr(scope, "reframed_query", "") or ""

    search = pubmed_search_tool(
        state["query"],
        pubmed_query=state["pubmed_query"],
        top_n=int(state["top_n"]),
        persist_dir=str(state["persist_dir"]),
        log_pipeline=bool(state.get("log_pipeline", False)),
        compute_device=state.get("compute_device"),
    )
    state["pmids"] = [str(item) for item in search.get("pmids", [])]
    state["records"] = list(search.get("records", []))
    state["docs_preview"] = list(search.get("docs_preview", []))
    state["context_top_k"] = int(search.get("context_top_k", state["top_n"]) or state["top_n"])

    retrieve = retriever_tool(
        abstract_store=search["abstract_store"],
        retrieval_query=state["retrieval_query"],
        top_n=int(state["top_n"]),
        use_reranker=bool(state.get("use_reranker", False)),
        log_pipeline=bool(state.get("log_pipeline", False)),
    )
    docs = list(retrieve.get("docs", []))
    state["reranker_active"] = bool(retrieve.get("reranker_active", False))
    state["retrieved_contexts"] = context_export_tool(docs, top_n=int(state["top_n"]))
    state["sources"] = citation_formatting_tool(docs, top_n=int(state["top_n"]))

    synthesis = answer_synthesis_tool(
        query=state["query"],
        retrieval_query=state["retrieval_query"],
        session_id=state["session_id"],
        branch_id=str(state.get("branch_id", "main") or "main"),
        llm=state.get("llm"),
        retriever=retrieve["retriever"],
        context_top_k=int(state.get("context_top_k", state["top_n"]) or state["top_n"]),
    )
    state["answer"] = str(synthesis.get("answer", "") or "")
    state["status"] = "answered"
    return state


@lru_cache(maxsize=1)
def _get_langgraph_runner() -> Callable[[AgentState], AgentState] | None:
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return None

    def guard_node(state: AgentState) -> AgentState:
        guard = safety_guardrail_tool(
            state["query"],
            session_id=state["session_id"],
            llm=state.get("llm"),
            log_pipeline=bool(state.get("log_pipeline", False)),
        )
        scope = guard.get("scope")
        updates: AgentState = {
            "status": str(guard.get("status", "out_of_scope")),
            "intent_label": str(guard.get("intent_label", "medical")),
            "intent_confidence": float(guard.get("intent_confidence", 0.0) or 0.0),
            "scope_label": getattr(scope, "label", ""),
            "scope_message": getattr(scope, "user_message", str(guard.get("message", "") or "")),
            "reframed_query": getattr(scope, "reframed_query", "") or "",
        }
        if not bool(guard.get("allow")):
            updates["answer"] = str(guard.get("message", "") or "")
        else:
            updates["scope"] = scope
        return updates

    def refine_node(state: AgentState) -> AgentState:
        refinement = query_refinement_tool(
            state["query"],
            scope=state["scope"],
            llm=state.get("llm"),
        )
        return {
            "pubmed_query": refinement["pubmed_query"],
            "retrieval_query": refinement["retrieval_query"],
            "reframe_note": refinement["reframe_note"],
        }

    def search_node(state: AgentState) -> AgentState:
        result = pubmed_search_tool(
            state["query"],
            pubmed_query=state["pubmed_query"],
            top_n=int(state["top_n"]),
            persist_dir=str(state["persist_dir"]),
            log_pipeline=bool(state.get("log_pipeline", False)),
            compute_device=state.get("compute_device"),
        )
        return {
            "pmids": result["pmids"],
            "records": result["records"],
            "docs_preview": result["docs_preview"],
            "abstract_store": result["abstract_store"],
            "context_top_k": int(result.get("context_top_k", state["top_n"]) or state["top_n"]),
        }

    def retrieve_node(state: AgentState) -> AgentState:
        result = retriever_tool(
            abstract_store=state["abstract_store"],
            retrieval_query=state["retrieval_query"],
            top_n=int(state["top_n"]),
            use_reranker=bool(state.get("use_reranker", False)),
            log_pipeline=bool(state.get("log_pipeline", False)),
        )
        docs = list(result["docs"])
        return {
            "retriever": result["retriever"],
            "reranker_active": bool(result["reranker_active"]),
            "sources": citation_formatting_tool(docs, top_n=int(state["top_n"])),
            "retrieved_contexts": context_export_tool(docs, top_n=int(state["top_n"])),
        }

    def synthesize_node(state: AgentState) -> AgentState:
        result = answer_synthesis_tool(
            query=state["query"],
            retrieval_query=state["retrieval_query"],
            session_id=state["session_id"],
            branch_id=str(state.get("branch_id", "main") or "main"),
            llm=state.get("llm"),
            retriever=state["retriever"],
            context_top_k=int(state.get("context_top_k", state["top_n"]) or state["top_n"]),
        )
        return {"answer": str(result.get("answer", "") or ""), "status": "answered"}

    graph = StateGraph(dict)
    graph.add_node("guard", guard_node)
    graph.add_node("refine", refine_node)
    graph.add_node("search", search_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.set_entry_point("guard")
    graph.add_conditional_edges(
        "guard",
        lambda state: "stop" if state.get("status") in {"smalltalk", "out_of_scope"} else "go",
        {"stop": END, "go": "refine"},
    )
    graph.add_edge("refine", "search")
    graph.add_edge("search", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    compiled = graph.compile()

    def runner(initial_state: AgentState) -> AgentState:
        result = compiled.invoke(initial_state)
        if isinstance(result, dict):
            return result
        return dict(result or {})

    return runner


def _state_to_payload(
    state: AgentState,
    *,
    request_id: str | None,
    include_paper_links: bool,
) -> PipelineResponse:
    status = str(state.get("status", "out_of_scope"))
    if status in {"smalltalk", "out_of_scope"}:
        return {
            "status": status,
            "answer": str(state.get("answer", "") or ""),
            "query": str(state.get("query", "") or ""),
            "sources": [],
            "intent_label": str(state.get("intent_label", "medical")),
            "intent_confidence": float(state.get("intent_confidence", 0.0) or 0.0),
            "scope_label": str(state.get("scope_label", "") or ""),
            "scope_message": str(state.get("scope_message", "") or ""),
            "reframed_query": str(state.get("reframed_query", "") or ""),
            "retrieved_contexts": [],
            "branch_id": str(state.get("branch_id", "main") or "main"),
            "request_id": request_id or "",
            "timings": dict(state.get("timings", {}) or {}),
        }

    sources = _filter_source_links(
        list(state.get("sources", []) or []),
        include_paper_links=include_paper_links,
    )
    payload = {
        "status": "answered",
        "answer": str(state.get("answer", "") or ""),
        "query": str(state.get("query", "") or ""),
        "sources": sources,
        "docs_preview": list(state.get("docs_preview", []) or []),
        "pubmed_query": str(state.get("pubmed_query", "") or ""),
        "reranker_active": bool(state.get("reranker_active", False)),
        "scope_label": str(state.get("scope_label", "BIOMEDICAL")),
        "scope_message": str(state.get("scope_message", "ok")),
        "reframed_query": str(state.get("reframed_query", "") or ""),
        "reframe_note": str(state.get("reframe_note", "") or ""),
        "intent_label": str(state.get("intent_label", "medical")),
        "intent_confidence": float(state.get("intent_confidence", 0.0) or 0.0),
        "retrieved_contexts": list(state.get("retrieved_contexts", []) or []),
        "branch_id": str(state.get("branch_id", "main") or "main"),
        "request_id": request_id or "",
        "timings": dict(state.get("timings", {}) or {}),
    }
    _add_source_count_note(payload, requested_top_n=int(state.get("top_n", 10) or 10))
    return payload


def _add_source_count_note(payload: dict[str, Any], *, requested_top_n: int) -> None:
    sources = payload.get("sources", []) or []
    if not isinstance(sources, list):
        return
    if len(sources) >= max(1, int(requested_top_n)):
        payload.pop("source_count_note", None)
        return
    payload["source_count_note"] = f"Only {len(sources)} unique papers were available for this query."


def _filter_source_links(
    sources: list[dict[str, Any]],
    *,
    include_paper_links: bool,
) -> list[dict[str, Any]]:
    if include_paper_links:
        return [dict(item) for item in sources]
    filtered: list[dict[str, Any]] = []
    for item in sources:
        copy = dict(item)
        copy.pop("doi", None)
        copy.pop("pmcid", None)
        copy.pop("fulltext_url", None)
        filtered.append(copy)
    return filtered


def _run_optional_validation(
    *,
    config: Any,
    query: str,
    answer: str,
    contexts: list[dict[str, str]],
    sources: list[dict[str, Any]],
    compute_device: str | None = None,
) -> dict[str, Any]:
    if not getattr(config, "validator_enabled", False):
        return {}
    if not answer.strip():
        return {}

    context_rows = [str(item.get("context", "") or "") for item in contexts]
    context_text = "\n\n---\n\n".join(context_rows)
    source_pmids = [str(item.get("pmid", "")).strip() for item in sources if item.get("pmid")]
    result = validate_answer(
        user_query=query,
        answer=answer,
        context=context_text,
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
        retrieved_docs=None,
        device=compute_device,
    )
    if result.get("valid", True):
        return {}
    details = result.get("details") or {}
    issues = details.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    warning = "Some parts of the answer may not be fully supported by the retrieved abstracts."
    return {
        "validation_warning": warning,
        "validation_issues": [str(item) for item in issues if str(item).strip()],
        "validation_confidence": f"{float(result.get('score', 0.0) or 0.0):.3f}",
    }


def _build_cached_answer_payload(
    cached_answer: dict[str, Any],
    *,
    request_id: str,
    query: str,
    branch_id: str,
    lookup_ms: float,
) -> PipelineResponse:
    payload: PipelineResponse = dict(cached_answer.get("response_payload", {}) or {})
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
    timings = dict(payload.get("timings", {}) or {})
    timings["answer_cache_lookup_ms"] = round(lookup_ms, 3)
    timings["total_ms"] = round(lookup_ms, 3)
    payload["timings"] = timings
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


def _log_agent_request(
    *,
    config: Any,
    request_id: str,
    session_id: str,
    branch_id: str,
    query: str,
    payload: Mapping[str, Any],
    started_at: float,
) -> None:
    timings = payload.get("timings")
    timing_payload = dict(timings or {}) if isinstance(timings, dict) else {}
    log_event(
        "request.complete",
        request_id=request_id,
        session_id=session_id,
        branch_id=branch_id,
        query_hash=hash_query_text(query),
        status=str(payload.get("status", "")),
        provider="nvidia" if getattr(config, "nvidia_api_key", None) else None,
        cache_hit=bool(payload.get("cache_hit", False) or payload.get("answer_cache_hit", False)),
        pmid_count=len(payload.get("sources", []) or []),
        retrieval_ms=timing_payload.get("retrieval_ms"),
        llm_ms=timing_payload.get("llm_ms"),
        total_ms=round((perf_counter() - started_at) * 1000.0, 3),
        store_path=str(getattr(config, "metrics_store_path")) if getattr(config, "metrics_mode", False) else None,
    )


def _sanitize_top_n(top_n: int) -> int:
    try:
        parsed = int(top_n)
    except (TypeError, ValueError):
        return 10
    return max(1, min(10, parsed))


def _get_llm_safe() -> Any | None:
    try:
        return get_nvidia_llm()
    except Exception:
        return None
