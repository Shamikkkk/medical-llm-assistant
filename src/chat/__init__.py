from __future__ import annotations

from typing import Any

from src.chat.contextualize import contextualize_question, summarize_last_topic

__all__ = [
    "contextualize_question",
    "summarize_last_topic",
    "invoke_chat_request",
    "stream_chat_request",
]


def __getattr__(name: str) -> Any:
    if name in {"invoke_chat_request", "stream_chat_request"}:
        from src.chat.router import invoke_chat_request, stream_chat_request

        return {
            "invoke_chat_request": invoke_chat_request,
            "stream_chat_request": stream_chat_request,
        }[name]
    raise AttributeError(name)
