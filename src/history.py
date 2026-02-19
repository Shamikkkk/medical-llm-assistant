from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

_HISTORY_STORE: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _HISTORY_STORE:
        _HISTORY_STORE[session_id] = ChatMessageHistory()
    return _HISTORY_STORE[session_id]


def clear_session_history(session_id: str) -> None:
    history = _HISTORY_STORE.pop(session_id, None)
    if history is not None:
        history.clear()
