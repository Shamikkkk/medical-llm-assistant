from __future__ import annotations

from typing import Any, Generator

from src.agent.runtime import invoke_chat_with_mode, stream_chat_with_mode
from src.chat.contextualize import contextualize_question
from src.integrations.nvidia import get_nvidia_llm

def invoke_chat_request(
    *,
    query: str,
    session_id: str,
    top_n: int,
    agent_mode: bool,
    follow_up_mode: bool,
    chat_messages: list[dict[str, Any]],
    **_: Any,
) -> dict[str, Any]:
    llm = _get_llm_safe()
    effective_query, topic_summary, rewritten = contextualize_question(
        user_query=query,
        chat_messages=chat_messages,
        follow_up_mode=follow_up_mode,
        llm=llm,
    )
    payload = invoke_chat_with_mode(
        effective_query,
        session_id=session_id,
        top_n=top_n,
        agent_mode=agent_mode,
    )
    payload = _drop_legacy_fields(payload)
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
    **_: Any,
) -> Generator[str, None, dict[str, Any]]:
    llm = _get_llm_safe()
    effective_query, topic_summary, rewritten = contextualize_question(
        user_query=query,
        chat_messages=chat_messages,
        follow_up_mode=follow_up_mode,
        llm=llm,
    )
    stream = stream_chat_with_mode(
        effective_query,
        session_id=session_id,
        top_n=top_n,
        agent_mode=agent_mode,
    )
    payload = yield from _yield_stream(stream)
    payload = _drop_legacy_fields(payload)
    payload.setdefault("query", query)
    payload["effective_query"] = effective_query
    payload["rewritten_query"] = effective_query if rewritten else ""
    payload["last_topic_summary"] = topic_summary
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
