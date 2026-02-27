from __future__ import annotations

from typing import Any, Generator, Mapping
import logging
import re

from src.core.chains import build_chat_chain, build_rag_chain
from src.core.config import load_config
from src.core.intent import classify_intent_details, smalltalk_reply
from src.core.scope import ScopeResult, classify_scope
from src.integrations.pubmed import (
    pubmed_efetch,
    pubmed_esearch,
    rewrite_to_pubmed_query,
    to_documents,
)
from src.integrations.storage import (
    get_abstract_store,
    get_query_cache_store,
    lookup_query_result_cache,
    remember_query_result,
    upsert_abstracts,
)
from src.logging_utils import log_llm_usage
from src.types import SourceItem

LOGGER = logging.getLogger("agent.tools")

_PERSONAL_ADVICE_PATTERNS = (
    r"\b(i am|i'm|my)\b",
    r"\bshould i\b",
    r"\bwhat should i take\b",
    r"\bdiagnose\b",
    r"\bprescribe\b",
    r"\bdosage for me\b",
)


def safety_guardrail_tool(
    query: str,
    *,
    session_id: str,
    llm: Any | None,
    log_pipeline: bool = False,
) -> dict[str, Any]:
    intent = classify_intent_details(query, llm=llm, log_enabled=log_pipeline)
    intent_label = str(intent.get("label", "")).lower()
    intent_confidence = float(intent.get("confidence", 0.0) or 0.0)
    if intent_label == "smalltalk":
        return {
            "allow": False,
            "status": "smalltalk",
            "message": smalltalk_reply(query, llm=llm),
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "scope": None,
        }

    scope = classify_scope(query, session_id=session_id, llm=llm)
    if not scope.allow:
        return {
            "allow": False,
            "status": "out_of_scope",
            "message": scope.user_message,
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "scope": scope,
        }

    if _is_personal_medical_advice_request(query):
        refusal = (
            "I can only provide academic literature summaries and cannot give personal "
            "medical advice, diagnosis, or treatment decisions. Please consult a licensed "
            "healthcare professional."
        )
        return {
            "allow": False,
            "status": "out_of_scope",
            "message": refusal,
            "intent_label": intent_label,
            "intent_confidence": intent_confidence,
            "scope": scope,
        }

    return {
        "allow": True,
        "status": "allowed",
        "message": "ok",
        "intent_label": intent_label,
        "intent_confidence": intent_confidence,
        "scope": scope,
    }


def query_refinement_tool(
    query: str,
    *,
    scope: ScopeResult,
    llm: Any | None,
) -> dict[str, str]:
    if scope.reframed_query:
        return {
            "pubmed_query": scope.reframed_query,
            "retrieval_query": scope.reframed_query,
            "reframe_note": scope.user_message if scope.user_message != "ok" else "",
        }

    pubmed_query = rewrite_to_pubmed_query(query, llm)
    retrieval_query = pubmed_query or query
    return {
        "pubmed_query": pubmed_query,
        "retrieval_query": retrieval_query,
        "reframe_note": "",
    }


def pubmed_search_tool(
    query: str,
    *,
    pubmed_query: str,
    top_n: int,
    persist_dir: str,
    log_pipeline: bool = False,
) -> dict[str, Any]:
    config = load_config()
    safe_top_n = min(_sanitize_top_n(top_n), int(config.max_abstracts))
    cache_store = get_query_cache_store(persist_dir)
    abstract_store = get_abstract_store(persist_dir)

    cached = lookup_query_result_cache(
        query,
        store=cache_store,
        ttl_seconds=int(config.pubmed_cache_ttl_seconds),
        negative_ttl_seconds=int(config.pubmed_negative_cache_ttl_seconds),
    )
    use_cache = False
    if cached:
        cached_query = str(cached.get("pubmed_query", "") or "")
        if not cached_query or cached_query == pubmed_query:
            use_cache = True

    if use_cache:
        effective_pubmed_query = str(cached.get("pubmed_query") or pubmed_query or query)
        pmids = [str(item) for item in (cached.get("pmids") or [])][:safe_top_n]
        records = pubmed_efetch(pmids) if pmids else []
        documents = to_documents(records)
        cache_status = "hit"
    else:
        effective_pubmed_query = pubmed_query or query
        pmids = pubmed_esearch(effective_pubmed_query, retmax=safe_top_n)
        records = pubmed_efetch(pmids)
        documents = to_documents(records)
        remember_query_result(
            query,
            pubmed_query=effective_pubmed_query,
            pmids=pmids,
            store=cache_store,
        )
        cache_status = "miss"

    embedded_count = upsert_abstracts(
        abstract_store,
        documents,
        query_text=query,
        pubmed_query=effective_pubmed_query,
        log_pipeline=log_pipeline,
    )
    if log_pipeline:
        LOGGER.info(
            "[AGENT] PubMed search | cache=%s pmids=%s records=%s embedded=%s",
            cache_status,
            len(pmids),
            len(records),
            embedded_count,
        )

    return {
        "cache_status": cache_status,
        "pubmed_query": effective_pubmed_query,
        "pmids": pmids,
        "records": records,
        "documents": documents,
        "docs_preview": _build_docs_preview(records, top_n=safe_top_n),
        "abstract_store": abstract_store,
        "embedded_count": embedded_count,
    }


def retriever_tool(
    *,
    abstract_store: Any,
    retrieval_query: str,
    top_n: int,
    use_reranker: bool,
    log_pipeline: bool = False,
) -> dict[str, Any]:
    safe_top_n = _sanitize_top_n(top_n)
    retriever = abstract_store.as_retriever(search_kwargs={"k": safe_top_n})
    reranker_active = False
    if use_reranker:
        try:
            from langchain.retrievers import ContextualCompressionRetriever
            from langchain_community.document_compressors import FlashrankRerank

            reranker = ContextualCompressionRetriever(
                base_compressor=FlashrankRerank(),
                base_retriever=retriever,
            )
            retriever = reranker
            reranker_active = True
        except Exception:
            reranker_active = False

    try:
        docs = retriever.invoke(retrieval_query)
    except Exception:
        docs = []

    docs_list = list(docs or []) if not isinstance(docs, list) else docs
    if log_pipeline:
        LOGGER.info(
            "[AGENT] Retriever | k=%s reranker_active=%s docs=%s",
            safe_top_n,
            reranker_active,
            len(docs_list),
        )
    return {
        "retriever": retriever,
        "docs": docs_list,
        "reranker_active": reranker_active,
    }


def answer_synthesis_tool(
    *,
    query: str,
    retrieval_query: str,
    session_id: str,
    llm: Any | None,
    retriever: Any,
) -> dict[str, Any]:
    if llm is None:
        return {
            "answer": "LLM not configured. Set NVIDIA_API_KEY to enable contextual answers.",
            "raw": None,
        }

    base_chain = build_rag_chain(llm, retriever)
    chat_chain = build_chat_chain(base_chain)
    raw = chat_chain.invoke(
        {"input": query, "retrieval_query": retrieval_query},
        config={"configurable": {"session_id": session_id}},
    )
    log_llm_usage("agent.answer.invoke", raw)
    return {"answer": _extract_message_text(raw), "raw": raw}


def answer_synthesis_stream_tool(
    *,
    query: str,
    retrieval_query: str,
    session_id: str,
    llm: Any | None,
    retriever: Any,
) -> Generator[str, None, dict[str, Any]]:
    if llm is None:
        fallback = "LLM not configured. Set NVIDIA_API_KEY to enable contextual answers."
        yield fallback
        return {"answer": fallback, "raw": None}

    base_chain = build_rag_chain(llm, retriever)
    chat_chain = build_chat_chain(base_chain)
    usage_candidate: Any | None = None
    answer_text = ""

    for chunk in chat_chain.stream(
        {"input": query, "retrieval_query": retrieval_query},
        config={"configurable": {"session_id": session_id}},
    ):
        if _has_usage_metadata(chunk):
            usage_candidate = chunk
        text = _extract_message_text(chunk)
        if not text:
            continue
        answer_text += text
        yield text

    log_llm_usage("agent.answer.stream", usage_candidate)
    return {"answer": answer_text, "raw": usage_candidate}


def citation_formatting_tool(docs: list[Any], *, top_n: int) -> list[SourceItem]:
    safe_top_n = _sanitize_top_n(top_n)
    sources: list[SourceItem] = []
    seen_pmids: set[str] = set()
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        pmid = str(meta.get("pmid", "") or "").strip()
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
        if len(sources) >= safe_top_n:
            break
    return sources


def context_export_tool(docs: list[Any], *, top_n: int) -> list[dict[str, str]]:
    safe_top_n = _sanitize_top_n(top_n)
    rows: list[dict[str, str]] = []
    seen_pmids: set[str] = set()
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        pmid = str(meta.get("pmid", "") or "").strip()
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        rows.append(
            {
                "pmid": pmid,
                "title": str(meta.get("title", "") or ""),
                "journal": str(meta.get("journal", "") or ""),
                "year": str(meta.get("year", "") or ""),
                "context": str(getattr(doc, "page_content", "") or "")[:4000],
            }
        )
        if len(rows) >= safe_top_n:
            break
    return rows


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


def _sanitize_top_n(top_n: int) -> int:
    try:
        parsed = int(top_n)
    except (TypeError, ValueError):
        return 10
    return max(1, min(10, parsed))


def _build_docs_preview(records: list[dict[str, Any]], *, top_n: int) -> list[SourceItem]:
    preview: list[SourceItem] = []
    seen: set[str] = set()
    for record in records:
        pmid = str(record.get("pmid", "") or "").strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        preview.append(
            {
                "rank": len(preview) + 1,
                "pmid": pmid,
                "title": str(record.get("title", "") or ""),
                "journal": str(record.get("journal", "") or ""),
                "year": str(record.get("year", "") or ""),
            }
        )
        if len(preview) >= top_n:
            break
    return preview


def _is_personal_medical_advice_request(query: str) -> bool:
    normalized = str(query or "").lower().strip()
    if not normalized:
        return False
    for pattern in _PERSONAL_ADVICE_PATTERNS:
        if re.search(pattern, normalized):
            return True
    return False
