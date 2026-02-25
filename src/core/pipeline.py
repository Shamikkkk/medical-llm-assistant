from __future__ import annotations

from typing import Any, Dict, List, Mapping
import logging
import re

from src.core.chains import _format_docs, build_chat_chain, build_rag_chain
from src.core.config import load_config
from src.intent import classify_intent_details, smalltalk_reply
from src.core.scope import ScopeResult, classify_scope
from src.history import get_session_history
from src.integrations.nvidia import get_nvidia_llm
from src.integrations.pubmed import (
    pubmed_efetch,
    pubmed_esearch,
    rewrite_to_pubmed_query,
    to_documents,
)
from src.integrations.storage import (
    add_query_cache_entry,
    get_abstract_store,
    get_query_cache_store,
    lookup_cached_query,
    upsert_abstracts,
)
from src.logging_utils import log_llm_usage
from src.types import PipelineResponse, SourceItem
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


def run_pipeline(query: str) -> PipelineResponse:
    """Backwards-compatible wrapper for single-turn usage."""
    return invoke_chat(query, session_id="default", top_n=10)


def invoke_chat(query: str, session_id: str, top_n: int = 10) -> PipelineResponse:
    safe_top_n = _sanitize_top_n(top_n)
    config = load_config()
    llm = _get_llm_safe()
    intent = classify_intent_details(query, llm, log_enabled=config.log_pipeline)
    intent_label = str(intent.get("label", "")).lower()
    intent_confidence = float(intent.get("confidence", 0.0) or 0.0)
    if intent_label == "smalltalk":
        return {
            "status": "smalltalk",
            "answer": smalltalk_reply(query, llm=llm),
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
        }

    scope = classify_scope(query, session_id=session_id, llm=llm)
    if not scope.allow:
        return {
            "status": "out_of_scope",
            "message": scope.user_message,
            "query": query,
            "scope_label": scope.label,
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
        }

    context = _prepare_chat_context(
        query=query,
        session_id=session_id,
        top_n=safe_top_n,
        llm=llm,
        scope=scope,
        config=config,
    )
    llm = context["llm"]

    if llm is None:
        return {
            "status": "answered",
            "answer": "LLM not configured. Set NVIDIA_API_KEY to enable contextual answers.",
            "docs_preview": context["docs_preview"],
            "pubmed_query": context["pubmed_query"],
            "scope_label": scope.label,
            "scope_message": scope.user_message,
            "reframed_query": scope.reframed_query or "",
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
        }

    base_chain = build_rag_chain(llm, context["retriever"])
    chat_chain = build_chat_chain(base_chain)

    raw_answer = chat_chain.invoke(
        {
            "input": query,
            "retrieval_query": context["retrieval_query"],
        },
        config={"configurable": {"session_id": session_id}},
    )
    log_llm_usage("answer.invoke", raw_answer)
    answer = _extract_message_text(raw_answer)

    retrieved_docs = _retrieve_docs_for_sources(
        retriever=context["retriever"],
        retrieval_query=context["retrieval_query"],
    )
    sources = _collect_sources_from_docs(retrieved_docs, top_n=safe_top_n)
    retrieved_contexts = _docs_to_eval_contexts(retrieved_docs, top_n=safe_top_n)
    validation_payload = _run_optional_validation(
        config=config,
        user_query=query,
        answer=answer,
        retrieved_docs=retrieved_docs,
        sources=sources,
    )

    payload: PipelineResponse = {
        "status": "answered",
        "answer": answer,
        "sources": sources,
        "reranker_active": context["reranker_active"],
        "pubmed_query": context["pubmed_query"],
        "docs_preview": context["docs_preview"],
        "scope_label": scope.label,
        "scope_message": scope.user_message,
        "reframed_query": scope.reframed_query or "",
        "reframe_note": context.get("reframe_note", ""),
        "query": query,
        "intent_label": intent_label,
        "intent_confidence": intent_confidence,
        "retrieved_contexts": retrieved_contexts,
    }
    payload.update(validation_payload)
    return payload


def stream_chat(query: str, session_id: str, top_n: int = 10):
    safe_top_n = _sanitize_top_n(top_n)
    config = load_config()
    llm = _get_llm_safe()
    intent = classify_intent_details(query, llm, log_enabled=config.log_pipeline)
    intent_label = str(intent.get("label", "")).lower()
    intent_confidence = float(intent.get("confidence", 0.0) or 0.0)
    if intent_label == "smalltalk":
        reply = smalltalk_reply(query, llm=llm)
        yield reply
        return {
            "status": "smalltalk",
            "answer": reply,
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
        }

    scope = classify_scope(query, session_id=session_id, llm=llm)
    if not scope.allow:
        yield scope.user_message
        return {
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
        }

    context = _prepare_chat_context(
        query=query,
        session_id=session_id,
        top_n=safe_top_n,
        llm=llm,
        scope=scope,
        config=config,
    )
    llm = context["llm"]

    if llm is None:
        fallback = "LLM not configured. Set NVIDIA_API_KEY to enable contextual answers."
        yield fallback
        return {
            "status": "answered",
            "answer": fallback,
            "docs_preview": context["docs_preview"],
            "pubmed_query": context["pubmed_query"],
            "scope_label": scope.label,
            "scope_message": scope.user_message,
            "reframed_query": scope.reframed_query or "",
            "query": query,
            "sources": [],
            "retrieved_contexts": [],
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
        }

    base_chain = build_rag_chain(llm, context["retriever"])
    chat_chain = build_chat_chain(base_chain)

    usage_candidate: Any | None = None
    answer_text = ""
    for chunk in chat_chain.stream(
        {
            "input": query,
            "retrieval_query": context["retrieval_query"],
        },
        config={"configurable": {"session_id": session_id}},
    ):
        if _has_usage_metadata(chunk):
            usage_candidate = chunk
        text = _coerce_stream_chunk(chunk)
        if text:
            answer_text += text
            yield text
    log_llm_usage("answer.stream", usage_candidate)

    retrieved_docs = _retrieve_docs_for_sources(
        retriever=context["retriever"],
        retrieval_query=context["retrieval_query"],
    )
    sources = _collect_sources_from_docs(retrieved_docs, top_n=safe_top_n)
    retrieved_contexts = _docs_to_eval_contexts(retrieved_docs, top_n=safe_top_n)
    validation_payload = _run_optional_validation(
        config=config,
        user_query=query,
        answer=answer_text,
        retrieved_docs=retrieved_docs,
        sources=sources,
    )
    payload: PipelineResponse = {
        "status": "answered",
        "answer": answer_text,
        "sources": sources,
        "reranker_active": context["reranker_active"],
        "pubmed_query": context["pubmed_query"],
        "docs_preview": context["docs_preview"],
        "scope_label": scope.label,
        "scope_message": scope.user_message,
        "reframed_query": scope.reframed_query or "",
        "reframe_note": context.get("reframe_note", ""),
        "query": query,
        "intent_label": intent_label,
        "intent_confidence": intent_confidence,
        "retrieved_contexts": retrieved_contexts,
    }
    payload.update(validation_payload)
    return payload


def build_contextual_retrieval_query(
    user_query: str,
    session_id: str,
    llm: Any | None = None,
    base_query: str | None = None,
) -> str:
    base_query = base_query or user_query
    normalized = " ".join(user_query.lower().strip().split())
    if not normalized:
        return base_query

    if not _is_followup_query(normalized):
        return base_query

    history_excerpt = _get_history_excerpt(session_id)
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
            cleaned = _sanitize_query(rewritten, fallback=base_query)
            return cleaned

    snippet = _get_last_assistant_excerpt(session_id)
    if snippet:
        expanded = f"{base_query} {snippet}"
        return expanded[:300].strip()
    return base_query


def _prepare_chat_context(
    query: str,
    session_id: str,
    top_n: int,
    llm: Any | None = None,
    scope: ScopeResult | None = None,
    config: Any | None = None,
) -> Dict[str, Any]:
    config = config or load_config()
    persist_dir = str(config.data_dir / "chroma")
    cache_store = get_query_cache_store(persist_dir)
    abstract_store = get_abstract_store(persist_dir)
    log_pipeline = bool(getattr(config, "log_pipeline", False))

    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Query start | query='%s' top_n=%s session_id=%s",
        _trim_text(query),
        top_n,
        session_id,
    )

    llm = llm or _get_llm_safe()
    scope = scope or ScopeResult(
        label="BIOMEDICAL",
        allow=True,
        user_message="ok",
        reframed_query=None,
        reason="default",
    )

    pubmed_query = scope.reframed_query or query
    pmids: List[str] = []
    records: List[Dict[str, Any]] = []
    documents = []

    cached = lookup_cached_query(cache_store, query)
    use_cache = False
    if cached and scope.reframed_query is None:
        use_cache = True
    elif cached and scope.reframed_query:
        cached_query = cached.get("pubmed_query") or ""
        if cached_query and cached_query == scope.reframed_query:
            use_cache = True

    if use_cache:
        pubmed_query = cached.get("pubmed_query") or pubmed_query
        pmids = [str(pmid) for pmid in (cached.get("pmids") or [])][:top_n]
        if pmids:
            records = pubmed_efetch(pmids)
            documents = to_documents(records)
        _pipeline_log(
            log_pipeline,
            "[PIPELINE] Cache HIT | query='%s' pubmed_query='%s' cached_pmids=%s records=%s documents=%s",
            _trim_text(query),
            _trim_text(pubmed_query),
            len(pmids),
            len(records),
            len(documents),
        )
    else:
        if scope.reframed_query:
            pubmed_query = scope.reframed_query
        else:
            pubmed_query = rewrite_to_pubmed_query(query, llm)
        pmids = pubmed_esearch(pubmed_query, retmax=top_n)
        records = pubmed_efetch(pmids)
        documents = to_documents(records)
        add_query_cache_entry(cache_store, query, pubmed_query=pubmed_query, pmids=pmids)
        _pipeline_log(
            log_pipeline,
            "[PIPELINE] Cache MISS | query='%s' pubmed_query='%s' esearch_pmids=%s efetch_records=%s documents=%s",
            _trim_text(query),
            _trim_text(pubmed_query),
            len(pmids),
            len(records),
            len(documents),
        )

    embedded_count = 0
    if documents:
        embedded_count = upsert_abstracts(
            abstract_store,
            documents,
            query_text=query,
            pubmed_query=pubmed_query,
            log_pipeline=log_pipeline,
        )
    else:
        _pipeline_log(
            log_pipeline,
            "[PIPELINE] Embedding skipped | reason=no_documents query='%s'",
            _trim_text(query),
        )

    if scope.reframed_query:
        retrieval_query = scope.reframed_query
    else:
        retrieval_query = build_contextual_retrieval_query(
            query, session_id, llm=llm, base_query=pubmed_query
        )

    retriever, reranker_active = _build_retriever(
        abstract_store=abstract_store,
        top_n=top_n,
        use_reranker=config.use_reranker,
        log_pipeline=log_pipeline,
    )

    reframe_note = ""
    if scope.reframed_query and scope.user_message != "ok":
        reframe_note = scope.user_message

    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Retrieval config | retriever_k=%s reranker_active=%s abstracts_fetched=%s abstracts_embedded=%s",
        top_n,
        reranker_active,
        len(records),
        embedded_count,
    )

    return {
        "llm": llm,
        "retriever": retriever,
        "reranker_active": reranker_active,
        "pubmed_query": pubmed_query,
        "retrieval_query": retrieval_query,
        "reframe_note": reframe_note,
        "docs_preview": _build_docs_preview(records, top_n=top_n),
        "cache_status": "hit" if use_cache else "miss",
        "abstracts_fetched": len(records),
        "abstracts_embedded": embedded_count,
    }


def _build_docs_preview(records: List[Dict[str, Any]], top_n: int) -> List[SourceItem]:
    items: list[SourceItem] = []
    seen: set[str] = set()
    for record in records:
        pmid = str(record.get("pmid", "")).strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        items.append(
            {
                "rank": len(items) + 1,
                "pmid": pmid,
                "title": str(record.get("title", "") or ""),
                "year": str(record.get("year", "") or ""),
                "journal": str(record.get("journal", "") or ""),
                "doi": str(record.get("doi", "") or ""),
                "pmcid": str(record.get("pmcid", "") or ""),
                "fulltext_url": str(record.get("fulltext_url", "") or ""),
            }
        )
        if len(items) >= top_n:
            break
    return items


def _collect_sources(retriever, retrieval_query: str, top_n: int) -> List[SourceItem]:
    docs = _retrieve_docs_for_sources(retriever=retriever, retrieval_query=retrieval_query)
    return _collect_sources_from_docs(docs, top_n=top_n)


def _retrieve_docs_for_sources(retriever, retrieval_query: str) -> list:
    try:
        docs = retriever.invoke(retrieval_query)
    except Exception:
        docs = []
    if not isinstance(docs, list):
        return list(docs or [])
    return docs


def _collect_sources_from_docs(docs: list, top_n: int) -> List[SourceItem]:
    sources: list[SourceItem] = []
    seen_pmids: set[str] = set()
    for doc in docs:
        meta = doc.metadata or {}
        pmid = str(meta.get("pmid", "")).strip()
        if not pmid or pmid in seen_pmids:
            continue
        sources.append(
            {
                "rank": len(sources) + 1,
                "pmid": pmid,
                "title": str(meta.get("title", "") or ""),
                "journal": str(meta.get("journal", "") or ""),
                "year": str(meta.get("year", "") or ""),
                "doi": str(meta.get("doi", "") or ""),
                "pmcid": str(meta.get("pmcid", "") or ""),
                "fulltext_url": str(meta.get("fulltext_url", "") or ""),
            }
        )
        seen_pmids.add(pmid)
        if len(sources) >= top_n:
            break
    return sources


def _docs_to_eval_contexts(docs: list, top_n: int) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    seen_pmids: set[str] = set()
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        pmid = str(metadata.get("pmid", "") or "").strip()
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        contexts.append(
            {
                "pmid": pmid,
                "title": str(metadata.get("title", "") or ""),
                "journal": str(metadata.get("journal", "") or ""),
                "year": str(metadata.get("year", "") or ""),
                "context": str(getattr(doc, "page_content", "") or "")[:4000],
            }
        )
        if len(contexts) >= top_n:
            break
    return contexts


def _build_retriever(abstract_store, top_n: int, use_reranker: bool, log_pipeline: bool = False):
    base_retriever = abstract_store.as_retriever(search_kwargs={"k": top_n})
    _pipeline_log(
        log_pipeline,
        "[PIPELINE] Built base retriever with k=%s",
        top_n,
    )
    if not use_reranker:
        return base_retriever, False

    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_community.document_compressors import FlashrankRerank

        compressor = FlashrankRerank()
        reranker = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        _pipeline_log(log_pipeline, "[PIPELINE] Reranker active: FlashrankRerank")
        return reranker, True
    except Exception:
        _pipeline_log(log_pipeline, "[PIPELINE] Reranker unavailable; falling back to base retriever")
        return base_retriever, False


def _sanitize_top_n(top_n: int) -> int:
    try:
        value = int(top_n)
    except (TypeError, ValueError):
        return 10
    return max(1, min(10, value))


def _get_llm_safe():
    try:
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


def _get_history_excerpt(session_id: str, max_messages: int = 6, max_chars: int = 1000) -> str:
    try:
        history = get_session_history(session_id)
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


def _get_last_assistant_excerpt(session_id: str, max_chars: int = 240) -> str:
    try:
        history = get_session_history(session_id)
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
    config: Any,
    user_query: str,
    answer: str,
    retrieved_docs: list,
    sources: list[SourceItem],
) -> Dict[str, Any]:
    if not getattr(config, "validator_enabled", False):
        return {}
    if not answer.strip():
        return {}

    context = _format_docs(retrieved_docs) if retrieved_docs else ""
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


def _pipeline_log(enabled: bool, message: str, *args: Any) -> None:
    if not enabled:
        return
    LOGGER.info(message, *args)


def _trim_text(text: str, limit: int = 140) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."
