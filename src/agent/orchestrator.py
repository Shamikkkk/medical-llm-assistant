from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable
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
from src.types import PipelineResponse
from src.validators import validate_answer

LOGGER = logging.getLogger("agent.orchestrator")


def invoke_agent_chat(query: str, session_id: str, top_n: int = 10) -> PipelineResponse:
    config = load_config()
    llm = _get_llm_safe()
    safe_top_n = _sanitize_top_n(top_n)
    initial: AgentState = {
        "query": query,
        "session_id": session_id,
        "top_n": safe_top_n,
        "llm": llm,
        "persist_dir": str(config.data_dir / "chroma"),
        "use_reranker": bool(config.use_reranker),
        "log_pipeline": bool(config.log_pipeline),
    }

    state = _run_agent(initial, use_langgraph=bool(config.agent_use_langgraph))
    payload = _state_to_payload(state)
    payload.update(
        _run_optional_validation(
            config=config,
            query=query,
            answer=str(payload.get("answer", "") or ""),
            contexts=state.get("retrieved_contexts", []) or [],
            sources=payload.get("sources", []) or [],
        )
    )
    return payload


def stream_agent_chat(query: str, session_id: str, top_n: int = 10):
    config = load_config()
    llm = _get_llm_safe()
    safe_top_n = _sanitize_top_n(top_n)
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
    docs_preview = search["docs_preview"]

    stream = answer_synthesis_stream_tool(
        query=query,
        retrieval_query=retrieval_query,
        session_id=session_id,
        llm=llm,
        retriever=retrieve["retriever"],
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
    }
    payload.update(
        _run_optional_validation(
            config=config,
            query=query,
            answer=answer_text,
            contexts=contexts,
            sources=sources,
        )
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
    )
    state["pmids"] = [str(item) for item in search.get("pmids", [])]
    state["records"] = list(search.get("records", []))
    state["docs_preview"] = list(search.get("docs_preview", []))

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
        llm=state.get("llm"),
        retriever=retrieve["retriever"],
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
        )
        return {
            "pmids": result["pmids"],
            "records": result["records"],
            "docs_preview": result["docs_preview"],
            "abstract_store": result["abstract_store"],
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
            llm=state.get("llm"),
            retriever=state["retriever"],
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


def _state_to_payload(state: AgentState) -> PipelineResponse:
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
        }

    return {
        "status": "answered",
        "answer": str(state.get("answer", "") or ""),
        "query": str(state.get("query", "") or ""),
        "sources": list(state.get("sources", []) or []),
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
    }


def _run_optional_validation(
    *,
    config: Any,
    query: str,
    answer: str,
    contexts: list[dict[str, str]],
    sources: list[dict[str, Any]],
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
