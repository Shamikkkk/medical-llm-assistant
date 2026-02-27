from __future__ import annotations

from typing import Any, Generator
from uuid import uuid4

from src.agent.runtime import invoke_chat_with_mode, stream_chat_with_mode
from src.chat.contextualize import contextualize_question
from src.integrations.nvidia import get_nvidia_llm
from src.logging_utils import hash_query_text, log_event

def invoke_chat_request(
    *,
    query: str,
    session_id: str,
    top_n: int,
    agent_mode: bool,
    follow_up_mode: bool,
    chat_messages: list[dict[str, Any]],
    show_papers: bool = True,
    conversation_summary: str = "",
    **_: Any,
) -> dict[str, Any]:
    request_id = uuid4().hex
    llm = _get_llm_safe()
    effective_query, topic_summary, rewritten = contextualize_question(
        user_query=query,
        chat_messages=chat_messages,
        follow_up_mode=follow_up_mode,
        conversation_summary=conversation_summary,
        llm=llm,
    )
    log_event(
        "request.start",
        request_id=request_id,
        session_id=session_id,
        query_hash=hash_query_text(query),
        effective_query_hash=hash_query_text(effective_query),
        agent_mode=agent_mode,
        follow_up_mode=follow_up_mode,
    )
    payload = invoke_chat_with_mode(
        effective_query,
        session_id=session_id,
        top_n=top_n,
        agent_mode=agent_mode,
        request_id=request_id,
        include_paper_links=show_papers,
    )
    payload = _drop_legacy_fields(payload)
    payload.setdefault("query", query)
    payload["effective_query"] = effective_query
    payload["rewritten_query"] = effective_query if rewritten else ""
    payload["last_topic_summary"] = topic_summary
    payload.setdefault("request_id", request_id)
    return payload


def stream_chat_request(
    *,
    query: str,
    session_id: str,
    top_n: int,
    agent_mode: bool,
    follow_up_mode: bool,
    chat_messages: list[dict[str, Any]],
    show_papers: bool = True,
    conversation_summary: str = "",
    **_: Any,
) -> Generator[str, None, dict[str, Any]]:
    request_id = uuid4().hex
    llm = _get_llm_safe()
    effective_query, topic_summary, rewritten = contextualize_question(
        user_query=query,
        chat_messages=chat_messages,
        follow_up_mode=follow_up_mode,
        conversation_summary=conversation_summary,
        llm=llm,
    )
    log_event(
        "request.start",
        request_id=request_id,
        session_id=session_id,
        query_hash=hash_query_text(query),
        effective_query_hash=hash_query_text(effective_query),
        agent_mode=agent_mode,
        follow_up_mode=follow_up_mode,
        streaming=True,
    )
    stream = stream_chat_with_mode(
        effective_query,
        session_id=session_id,
        top_n=top_n,
        agent_mode=agent_mode,
        request_id=request_id,
        include_paper_links=show_papers,
    )
    payload = yield from _yield_stream(stream)
    payload = _drop_legacy_fields(payload)
    payload.setdefault("query", query)
    payload["effective_query"] = effective_query
    payload["rewritten_query"] = effective_query if rewritten else ""
    payload["last_topic_summary"] = topic_summary
    payload.setdefault("request_id", request_id)
    return payload


def _yield_stream(stream) -> Generator[str, None, dict[str, Any]]:
    while True:
        try:
            chunk = next(stream)
        except StopIteration as exc:
            return dict(exc.value or {})
        if chunk:
            yield str(chunk)


def _get_llm_safe() -> Any | None:
    try:
        return get_nvidia_llm()
    except Exception:
        return None


def _drop_legacy_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = dict(payload or {})
    for key in list(cleaned.keys()):
        if str(key).startswith("paper_"):
            cleaned.pop(key, None)
    return cleaned
