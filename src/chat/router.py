from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Mapping
import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from src.agent.runtime import invoke_chat_with_mode, stream_chat_with_mode
from src.chat.contextualize import contextualize_question
from src.integrations.nvidia import get_nvidia_llm
from src.logging_utils import log_llm_usage
from src.papers.fetch import extract_text_from_uploaded_pdf, fetch_paper_content
from src.papers.fetch_fulltext import (
    STATUS_OK_OA_HTML,
    STATUS_PAYWALLED_OR_BLOCKED,
    STATUS_UNSUPPORTED_CONTENT,
    fetch_readable_text_from_url,
)
from src.papers.index import PaperIndexer
from src.papers.store import (
    PAPER_TIER_ABSTRACT,
    PAPER_TIER_FULL_TEXT,
    PAPER_TIER_UPLOADED_PDF,
    PaperContent,
    PaperStore,
    build_pubmed_url,
)

LOGGER = logging.getLogger("chat.router")

_BROADEN_RE = re.compile(
    r"\b(broaden|all papers|clear focus|outside this paper|general literature)\b",
    flags=re.IGNORECASE,
)

PAYWALL_NOTICE = (
    "Full text appears paywalled; I can't fetch it automatically. "
    "You can (a) paste sections, (b) upload the PDF, or (c) provide an "
    "accessible link that allows text extraction."
)


def invoke_chat_request(
    *,
    query: str,
    session_id: str,
    top_n: int,
    agent_mode: bool,
    follow_up_mode: bool,
    chat_messages: list[dict[str, Any]],
    selected_paper: dict[str, Any] | None,
    data_dir: str | Path,
) -> dict[str, Any]:
    llm = _get_llm_safe()
    effective_query, topic_summary, rewritten = contextualize_question(
        user_query=query,
        chat_messages=chat_messages,
        follow_up_mode=follow_up_mode,
        llm=llm,
    )
    if selected_paper and not _BROADEN_RE.search(query):
        payload = _invoke_paper_focus(
            query=effective_query,
            user_query=query,
            session_id=session_id,
            top_n=top_n,
            selected_paper=selected_paper,
            chat_messages=chat_messages,
            data_dir=data_dir,
            llm=llm,
        )
    else:
        payload = invoke_chat_with_mode(
            effective_query,
            session_id=session_id,
            top_n=top_n,
            agent_mode=agent_mode,
        )
        payload.setdefault("paper_focus_mode", False)

    payload.setdefault("query", query)
    payload["effective_query"] = effective_query
    payload["rewritten_query"] = effective_query if rewritten else ""
    payload["last_topic_summary"] = topic_summary
    return payload


def stream_chat_request(
    *,
    query: str,
    session_id: str,
    top_n: int,
    agent_mode: bool,
    follow_up_mode: bool,
    chat_messages: list[dict[str, Any]],
    selected_paper: dict[str, Any] | None,
    data_dir: str | Path,
) -> Generator[str, None, dict[str, Any]]:
    llm = _get_llm_safe()
    effective_query, topic_summary, rewritten = contextualize_question(
        user_query=query,
        chat_messages=chat_messages,
        follow_up_mode=follow_up_mode,
        llm=llm,
    )
    if selected_paper and not _BROADEN_RE.search(query):
        payload = yield from _stream_paper_focus(
            query=effective_query,
            user_query=query,
            session_id=session_id,
            top_n=top_n,
            selected_paper=selected_paper,
            chat_messages=chat_messages,
            data_dir=data_dir,
            llm=llm,
        )
    else:
        base_stream = stream_chat_with_mode(
            effective_query,
            session_id=session_id,
            top_n=top_n,
            agent_mode=agent_mode,
        )
        payload = yield from _yield_stream(base_stream)
        payload.setdefault("paper_focus_mode", False)

    payload.setdefault("query", query)
    payload["effective_query"] = effective_query
    payload["rewritten_query"] = effective_query if rewritten else ""
    payload["last_topic_summary"] = topic_summary
    return payload


def ingest_uploaded_pdf_for_selected_paper(
    *,
    selected_paper: dict[str, Any] | None,
    uploaded_bytes: bytes,
    file_name: str,
    data_dir: str | Path,
) -> dict[str, Any]:
    pmid = str((selected_paper or {}).get("pmid", "")).strip()
    if not pmid:
        return {"ok": False, "message": "Select a paper before uploading a PDF."}
    text = extract_text_from_uploaded_pdf(uploaded_bytes)
    if not text:
        return {"ok": False, "message": "Unable to extract text from PDF."}

    root = Path(data_dir).expanduser().absolute() / "papers"
    store = PaperStore(root)
    paper = store.load(pmid)
    if paper is None:
        fetched = fetch_paper_content(pmid)
        if fetched is None:
            return {"ok": False, "message": "Could not load paper metadata for selected PMID."}
        store.save(fetched)
        paper = fetched

    updated = store.update_with_pdf(pmid=pmid, pdf_text=text, file_name=file_name)
    if updated is None:
        return {"ok": False, "message": "Could not update paper store with PDF content."}
    indexer = PaperIndexer(root)
    indexed = indexer.index_paper(updated)
    return {
        "ok": True,
        "message": f"Uploaded PDF indexed with {indexed} chunks. Paper focus now uses uploaded full text.",
        "paper": {
            "pmid": updated.pmid,
            "title": updated.title,
            "journal": updated.journal,
            "year": updated.year,
            "doi": updated.doi,
            "pmcid": updated.pmcid,
            "fulltext_url": updated.fulltext_url,
            "content_tier": updated.content_tier,
        },
    }


def ingest_link_for_selected_paper(
    *,
    selected_paper: dict[str, Any] | None,
    link_url: str,
    data_dir: str | Path,
) -> dict[str, Any]:
    pmid = str((selected_paper or {}).get("pmid", "")).strip()
    if not pmid:
        return {"ok": False, "message": "Select a paper before ingesting from a link."}
    candidate_url = str(link_url or "").strip()
    if not candidate_url:
        return {"ok": False, "message": "Provide a publicly accessible paper URL first."}

    text, meta, status = fetch_readable_text_from_url(candidate_url)
    if status != STATUS_OK_OA_HTML or not text.strip():
        if status == STATUS_PAYWALLED_OR_BLOCKED:
            return {"ok": False, "status": status, "message": PAYWALL_NOTICE}
        if status == STATUS_UNSUPPORTED_CONTENT:
            return {
                "ok": False,
                "status": status,
                "message": (
                    "Link content is not readable as public article HTML. "
                    "Upload a PDF or provide another accessible URL."
                ),
            }
        return {
            "ok": False,
            "status": status,
            "message": "Could not fetch readable text from the provided link.",
        }

    root = Path(data_dir).expanduser().absolute() / "papers"
    store = PaperStore(root)
    paper = store.load(pmid)
    if paper is None:
        fetched = fetch_paper_content(pmid)
        if fetched is None:
            return {"ok": False, "message": "Could not load paper metadata for selected PMID."}
        store.save(fetched)
        paper = fetched

    final_url = str(meta.get("final_url") or candidate_url)
    updated = store.update_with_link_text(
        pmid=pmid,
        link_url=final_url,
        link_text=text,
        source_label="OA_HTML",
        notes="Full text extracted from a publicly accessible link.",
    )
    if updated is None:
        return {"ok": False, "message": "Could not store link-derived full text."}
    indexer = PaperIndexer(root)
    indexed = indexer.index_paper(updated)
    return {
        "ok": True,
        "status": status,
        "message": f"Fetched and indexed link content ({indexed} chunks). Paper focus uses this text.",
        "paper": {
            "pmid": updated.pmid,
            "title": updated.title,
            "journal": updated.journal,
            "year": updated.year,
            "doi": updated.doi,
            "fulltext_url": updated.fulltext_url or final_url,
            "content_tier": updated.content_tier,
        },
    }


def _invoke_paper_focus(
    *,
    query: str,
    user_query: str,
    session_id: str,
    top_n: int,
    selected_paper: dict[str, Any],
    chat_messages: list[dict[str, Any]],
    data_dir: str | Path,
    llm: Any | None,
) -> dict[str, Any]:
    result = _prepare_paper_context(
        selected_paper=selected_paper,
        data_dir=data_dir,
        query=query,
        top_n=top_n,
    )
    if not result["ok"]:
        message = result["message"]
        return {
            "status": "answered",
            "answer": message,
            "query": user_query,
            "sources": result.get("sources", []),
            "docs_preview": result.get("sources", []),
            "paper_focus_mode": True,
            "paper_focus_notice": message,
            "retrieved_contexts": [],
        }
    paper = result["paper"]
    docs = result["docs"]
    context_rows = result["contexts"]
    answer = _generate_paper_answer(
        query=query,
        session_id=session_id,
        llm=llm,
        paper=paper,
        docs=docs,
        chat_messages=chat_messages,
    )
    return {
        "status": "answered",
        "answer": answer,
        "query": user_query,
        "sources": result["sources"],
        "docs_preview": result["sources"],
        "pubmed_query": paper.title or query,
        "paper_focus_mode": True,
        "paper_focus_notice": _paper_notice(paper),
        "paper_focus_pmid": paper.pmid,
        "retrieved_contexts": context_rows,
    }


def _stream_paper_focus(
    *,
    query: str,
    user_query: str,
    session_id: str,
    top_n: int,
    selected_paper: dict[str, Any],
    chat_messages: list[dict[str, Any]],
    data_dir: str | Path,
    llm: Any | None,
) -> Generator[str, None, dict[str, Any]]:
    prepared = _prepare_paper_context(
        selected_paper=selected_paper,
        data_dir=data_dir,
        query=query,
        top_n=top_n,
    )
    if not prepared["ok"]:
        message = prepared["message"]
        yield message
        return {
            "status": "answered",
            "answer": message,
            "query": user_query,
            "sources": prepared.get("sources", []),
            "docs_preview": prepared.get("sources", []),
            "paper_focus_mode": True,
            "paper_focus_notice": message,
            "retrieved_contexts": [],
        }

    paper = prepared["paper"]
    docs = prepared["docs"]
    contexts = prepared["contexts"]
    if llm is None:
        fallback = "LLM not configured. Set NVIDIA_API_KEY to answer paper-focused follow-up questions."
        yield fallback
        return {
            "status": "answered",
            "answer": fallback,
            "query": user_query,
            "sources": prepared["sources"],
            "docs_preview": prepared["sources"],
            "paper_focus_mode": True,
            "paper_focus_notice": _paper_notice(paper),
            "paper_focus_pmid": paper.pmid,
            "retrieved_contexts": contexts,
        }

    prompt = _paper_prompt_template()
    chain = prompt | llm
    usage_candidate: Any | None = None
    answer_text = ""
    history_excerpt = _history_excerpt(chat_messages)
    context_text = _format_paper_docs(docs)
    for chunk in chain.stream(
        {
            "question": query,
            "pmid": paper.pmid,
            "title": paper.title,
            "tier_note": _paper_notice(paper),
            "history": history_excerpt,
            "context": context_text,
        }
    ):
        if _has_usage_metadata(chunk):
            usage_candidate = chunk
        text = _extract_text(chunk)
        if not text:
            continue
        answer_text += text
        yield text
    log_llm_usage("paper_focus.answer.stream", usage_candidate)

    if not answer_text.strip():
        answer_text = (
            f"I could not find enough supporting content in PMID {paper.pmid}. "
            "You can broaden the search beyond this paper or upload a PDF for richer context."
        )
    return {
        "status": "answered",
        "answer": answer_text,
        "query": user_query,
        "sources": prepared["sources"],
        "docs_preview": prepared["sources"],
        "pubmed_query": paper.title or query,
        "paper_focus_mode": True,
        "paper_focus_notice": _paper_notice(paper),
        "paper_focus_pmid": paper.pmid,
        "retrieved_contexts": contexts,
    }


def _prepare_paper_context(
    *,
    selected_paper: dict[str, Any],
    data_dir: str | Path,
    query: str,
    top_n: int,
) -> dict[str, Any]:
    pmid = str(selected_paper.get("pmid", "")).strip()
    title = str(selected_paper.get("title", "")).strip()
    journal = str(selected_paper.get("journal", "")).strip()
    year = str(selected_paper.get("year", "")).strip()
    if not pmid:
        return {"ok": False, "message": "No paper selected for paper focus mode.", "sources": []}

    root = Path(data_dir).expanduser().absolute() / "papers"
    store = PaperStore(root)
    paper = store.load(pmid)
    if paper is None:
        fetched = fetch_paper_content(pmid)
        if fetched is None:
            message = (
                f"Could not fetch metadata/content for PMID {pmid}. "
                "Try clearing paper focus and running a broader search."
            )
            return {
                "ok": False,
                "message": message,
                "sources": [
                    {
                        "rank": 1,
                        "pmid": pmid,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "doi": str(selected_paper.get("doi", "") or ""),
                        "pmcid": str(selected_paper.get("pmcid", "") or ""),
                        "fulltext_url": str(selected_paper.get("fulltext_url", "") or ""),
                    }
                ],
            }
        store.save(fetched)
        paper = fetched
    elif (
        paper.content_tier == PAPER_TIER_ABSTRACT
        and paper.fulltext_url
        and "paywalled" not in str(paper.notes or "").lower()
    ):
        link_text, link_meta, status = fetch_readable_text_from_url(paper.fulltext_url)
        if status == STATUS_OK_OA_HTML and link_text.strip():
            updated = store.update_with_link_text(
                pmid=paper.pmid,
                link_url=str(link_meta.get("final_url") or paper.fulltext_url),
                link_text=link_text,
                source_label="OA_HTML",
                notes="Full text extracted from a publicly accessible link.",
            )
            if updated is not None:
                paper = updated

    indexer = PaperIndexer(root)
    indexer.index_paper(paper)
    docs = indexer.retrieve(pmid, query, k=max(3, min(8, int(top_n))))
    source_item = {
        "rank": 1,
        "pmid": paper.pmid,
        "title": paper.title,
        "journal": paper.journal,
        "year": paper.year,
        "doi": paper.doi,
        "pmcid": paper.pmcid,
        "fulltext_url": paper.fulltext_url or build_pubmed_url(paper.pmid),
    }
    context_rows = [
        {
            "pmid": paper.pmid,
            "title": str((doc.metadata or {}).get("title", "") or paper.title),
            "journal": str((doc.metadata or {}).get("journal", "") or paper.journal),
            "year": str((doc.metadata or {}).get("year", "") or paper.year),
            "context": str(getattr(doc, "page_content", "") or "")[:4000],
        }
        for doc in docs
    ]
    if not docs:
        notice = (
            f"I could not retrieve enough indexed context for PMID {paper.pmid}. "
            "You can upload a PDF for this paper, ingest from an accessible link, "
            "or clear paper focus to broaden retrieval."
        )
        return {
            "ok": False,
            "message": notice,
            "sources": [source_item],
            "paper": paper,
        }
    return {
        "ok": True,
        "paper": paper,
        "docs": docs,
        "sources": [source_item],
        "contexts": context_rows,
    }


def _generate_paper_answer(
    *,
    query: str,
    session_id: str,
    llm: Any | None,
    paper: PaperContent,
    docs: list[Any],
    chat_messages: list[dict[str, Any]],
) -> str:
    del session_id
    if llm is None:
        return "LLM not configured. Set NVIDIA_API_KEY to enable paper-focused answers."
    prompt = _paper_prompt_template()
    chain = prompt | llm
    context_text = _format_paper_docs(docs)
    history_excerpt = _history_excerpt(chat_messages)
    raw = chain.invoke(
        {
            "question": query,
            "pmid": paper.pmid,
            "title": paper.title,
            "tier_note": _paper_notice(paper),
            "history": history_excerpt,
            "context": context_text,
        }
    )
    log_llm_usage("paper_focus.answer.invoke", raw)
    text = _extract_text(raw).strip()
    if not text:
        return (
            f"I could not find enough supporting content in PMID {paper.pmid}. "
            "You can broaden search beyond this paper or upload a PDF."
        )
    return text


def _paper_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a medical literature assistant for academic use only. "
                "Do NOT provide diagnosis, treatment instructions, or personal medical advice. "
                "Answer only using the provided paper context (PMC/OA/PDF/user-provided text). "
                "Do not claim to have read inaccessible paywalled full text. "
                "If evidence is insufficient in the context, say so clearly and suggest broadening retrieval "
                "or uploading the paper PDF. Include citations using format [PMID:{pmid}] for claims.",
            ),
            (
                "human",
                "Paper focus is active.\n"
                "PMID: {pmid}\n"
                "Title: {title}\n"
                "Content status: {tier_note}\n\n"
                "Conversation context (for reference only):\n{history}\n\n"
                "Question:\n{question}\n\n"
                "Paper context:\n{context}\n\n"
                "Provide an evidence-grounded answer with citations [PMID:{pmid}].",
            ),
        ]
    )


def _paper_notice(paper: PaperContent) -> str:
    if paper.content_tier == PAPER_TIER_UPLOADED_PDF:
        return "Using uploaded PDF content as primary source."
    if paper.content_tier == PAPER_TIER_FULL_TEXT:
        if paper.source_label == "OA_HTML":
            return "Using full text extracted from an open-access web page."
        return "Using available full text content."
    if paper.content_tier == PAPER_TIER_ABSTRACT:
        if paper.notes:
            return paper.notes
        return "Answer based on abstract/metadata only; full text not available."
    return "Using paper context currently available in the index."


def _history_excerpt(messages: list[dict[str, Any]], max_turns: int = 6) -> str:
    rows: list[str] = []
    for message in (messages or [])[-max_turns:]:
        role = str(message.get("role", ""))
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        speaker = "assistant" if role == "assistant" else "user"
        rows.append(f"{speaker}: {content}")
    return "\n".join(rows)


def _format_paper_docs(docs: list[Any]) -> str:
    sections: list[str] = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        pmid = str(metadata.get("pmid", "") or "")
        chunk_index = str(metadata.get("chunk_index", ""))
        title = str(metadata.get("title", "") or "")
        text = str(getattr(doc, "page_content", "") or "")
        sections.append(
            "PMID: {pmid}\nChunk: {chunk}\nTitle: {title}\nText: {text}".format(
                pmid=pmid,
                chunk=chunk_index,
                title=title,
                text=text,
            )
        )
    return "\n\n---\n\n".join(sections)


def _yield_stream(stream) -> Generator[str, None, dict[str, Any]]:
    while True:
        try:
            chunk = next(stream)
        except StopIteration as exc:
            return dict(exc.value or {})
        if chunk:
            yield str(chunk)


def _extract_text(payload: Any) -> str:
    content = getattr(payload, "content", None)
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
    return str(payload or "")


def _has_usage_metadata(response: Any) -> bool:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata:
        return True
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, Mapping) and (
        response_metadata.get("token_usage") or response_metadata.get("usage")
    ):
        return True
    return False


def _get_llm_safe() -> Any | None:
    try:
        return get_nvidia_llm()
    except Exception:
        return None
