from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import streamlit as st


def init_state(default_top_n: int = 10) -> None:
    legacy_messages = list(st.session_state.get("messages", []))
    legacy_session = str(st.session_state.get("session_id", "") or uuid4())

    if "chats" not in st.session_state:
        st.session_state.chats = [
            _new_chat_record(chat_id=legacy_session, messages=legacy_messages)
        ]
    else:
        for chat in st.session_state.chats:
            if "context_state" not in chat or not isinstance(chat.get("context_state"), dict):
                chat["context_state"] = _default_context_state()

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = st.session_state.chats[-1]["chat_id"]

    if _find_chat_index(st.session_state.active_chat_id) is None:
        st.session_state.active_chat_id = st.session_state.chats[-1]["chat_id"]

    if "top_n" not in st.session_state:
        st.session_state.top_n = _sanitize_top_n(default_top_n)
    else:
        st.session_state.top_n = _sanitize_top_n(st.session_state.top_n)

    _load_active_messages()
    st.session_state.session_id = st.session_state.active_chat_id


def get_active_chat_id() -> str:
    return str(st.session_state.active_chat_id)


def get_active_messages() -> list[dict]:
    return list(st.session_state.messages)


def append_active_message(message: dict) -> None:
    st.session_state.messages.append(message)
    _sync_active_messages()


def set_active_messages(messages: list[dict]) -> None:
    st.session_state.messages = list(messages)
    _sync_active_messages()


def clear_active_messages() -> None:
    st.session_state.messages = []
    st.session_state.context_state = _default_context_state()
    _sync_active_messages()


def new_chat() -> str:
    _sync_active_messages()
    record = _new_chat_record()
    st.session_state.chats.append(record)
    st.session_state.active_chat_id = record["chat_id"]
    st.session_state.session_id = record["chat_id"]
    st.session_state.messages = []
    st.session_state.context_state = _default_context_state()
    return record["chat_id"]


def switch_chat(chat_id: str) -> None:
    if chat_id == st.session_state.active_chat_id:
        return
    _sync_active_messages()
    st.session_state.active_chat_id = chat_id
    st.session_state.session_id = chat_id
    _load_active_messages()


def get_recent_chats(limit: int = 5) -> list[dict]:
    sorted_chats = sorted(
        st.session_state.chats,
        key=lambda item: str(item.get("created_at", "")),
        reverse=True,
    )
    return sorted_chats[: max(1, int(limit))]


def get_top_n() -> int:
    return _sanitize_top_n(st.session_state.get("top_n", 10))


def set_top_n(value: int) -> None:
    st.session_state.top_n = _sanitize_top_n(value)


def get_active_context_state() -> dict:
    state = st.session_state.get("context_state")
    if not isinstance(state, dict):
        state = _default_context_state()
        st.session_state.context_state = state
    return dict(state)


def update_active_context_state(**updates: Any) -> None:
    current = get_active_context_state()
    current.update(updates)
    st.session_state.context_state = current
    _sync_active_messages()


def set_follow_up_mode(enabled: bool) -> None:
    update_active_context_state(follow_up_mode=bool(enabled))


def get_follow_up_mode() -> bool:
    return bool(get_active_context_state().get("follow_up_mode", False))


def get_selected_paper() -> dict | None:
    value = get_active_context_state().get("selected_paper")
    if isinstance(value, dict):
        return dict(value)
    return None


def set_selected_paper(paper: dict | None) -> None:
    if paper and isinstance(paper, dict):
        payload = {
            "pmid": str(paper.get("pmid", "")).strip(),
            "title": str(paper.get("title", "")).strip(),
            "journal": str(paper.get("journal", "")).strip(),
            "year": str(paper.get("year", "")).strip(),
            "doi": str(paper.get("doi", "")).strip(),
            "pmcid": str(paper.get("pmcid", "")).strip(),
            "fulltext_url": str(paper.get("fulltext_url", "")).strip(),
        }
        update_active_context_state(selected_paper=payload)
        return
    update_active_context_state(selected_paper=None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_chat_record(chat_id: str | None = None, messages: list[dict] | None = None) -> dict:
    payload = list(messages or [])
    return {
        "chat_id": chat_id or str(uuid4()),
        "title": _derive_chat_title(payload),
        "created_at": _utc_now_iso(),
        "messages": payload,
        "context_state": _default_context_state(),
    }


def _derive_chat_title(messages: list[dict]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = str(message.get("content", "")).strip()
        if text:
            return text[:48] + ("..." if len(text) > 48 else "")
    return "New conversation"


def _find_chat_index(chat_id: str) -> int | None:
    for idx, chat in enumerate(st.session_state.chats):
        if chat.get("chat_id") == chat_id:
            return idx
    return None


def _load_active_messages() -> None:
    idx = _find_chat_index(st.session_state.active_chat_id)
    if idx is None:
        st.session_state.messages = []
        st.session_state.context_state = _default_context_state()
        return
    st.session_state.messages = list(st.session_state.chats[idx].get("messages", []))
    context_state = st.session_state.chats[idx].get("context_state")
    if not isinstance(context_state, dict):
        context_state = _default_context_state()
        st.session_state.chats[idx]["context_state"] = context_state
    st.session_state.context_state = dict(context_state)


def _sync_active_messages() -> None:
    active_id = str(st.session_state.active_chat_id)
    idx = _find_chat_index(active_id)
    if idx is None:
        st.session_state.chats.append(_new_chat_record(chat_id=active_id))
        idx = len(st.session_state.chats) - 1
    st.session_state.chats[idx]["messages"] = list(st.session_state.messages)
    st.session_state.chats[idx]["title"] = _derive_chat_title(st.session_state.messages)
    context_state = st.session_state.get("context_state")
    if not isinstance(context_state, dict):
        context_state = _default_context_state()
    st.session_state.chats[idx]["context_state"] = dict(context_state)


def _sanitize_top_n(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10
    return max(1, min(10, parsed))


def _default_context_state() -> dict:
    return {
        "follow_up_mode": False,
        "selected_paper": None,
        "last_topic_summary": "",
        "last_retrieved_sources": [],
    }
