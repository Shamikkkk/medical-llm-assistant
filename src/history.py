from __future__ import annotations

from typing import Iterable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

_HISTORY_STORE: dict[tuple[str, str], BaseChatMessageHistory] = {}


def get_session_history(session_id: str, branch_id: str = "main") -> BaseChatMessageHistory:
    key = _history_key(session_id, branch_id)
    if key not in _HISTORY_STORE:
        _HISTORY_STORE[key] = ChatMessageHistory()
    return _HISTORY_STORE[key]


def replace_session_history(
    session_id: str,
    branch_id: str = "main",
    messages: Iterable[dict] | None = None,
) -> BaseChatMessageHistory:
    history = ChatMessageHistory()
    for message in list(messages or []):
        role = str(message.get("role", "") or "").strip().lower()
        content = str(message.get("content", "") or "")
        if not content:
            continue
        if role == "assistant":
            history.add_message(AIMessage(content=content))
        else:
            history.add_message(HumanMessage(content=content))
    _HISTORY_STORE[_history_key(session_id, branch_id)] = history
    return history


def clear_session_history(session_id: str, branch_id: str | None = None) -> None:
    if branch_id is None:
        doomed = [
            key
            for key in _HISTORY_STORE
            if key[0] == str(session_id or "").strip()
        ]
        for key in doomed:
            history = _HISTORY_STORE.pop(key, None)
            if history is not None:
                history.clear()
        return

    history = _HISTORY_STORE.pop(_history_key(session_id, branch_id), None)
    if history is not None:
        history.clear()


def _history_key(session_id: str, branch_id: str) -> tuple[str, str]:
    return (
        str(session_id or "").strip() or "default",
        str(branch_id or "").strip() or "main",
    )
