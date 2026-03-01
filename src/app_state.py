from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import streamlit as st

MAIN_BRANCH_ID = "main"


def init_state(
    default_top_n: int = 10,
    *,
    default_show_papers: bool = False,
    default_show_rewritten_query: bool = False,
    default_auto_scroll: bool = True,
    default_follow_up_mode: bool = True,
    default_compute_device_preference: str = "auto",
) -> None:
    legacy_messages = list(st.session_state.get("messages", []))
    legacy_session = str(st.session_state.get("session_id", "") or uuid4())

    if "chats" not in st.session_state:
        st.session_state.chats = [
            _new_chat_record(
                chat_id=legacy_session,
                messages=legacy_messages,
                default_show_papers=default_show_papers,
                default_show_rewritten_query=default_show_rewritten_query,
                default_auto_scroll=default_auto_scroll,
                default_follow_up_mode=default_follow_up_mode,
            )
        ]
    else:
        st.session_state.chats = [
            _migrate_chat_record(
                chat,
                default_show_papers=default_show_papers,
                default_show_rewritten_query=default_show_rewritten_query,
                default_auto_scroll=default_auto_scroll,
                default_follow_up_mode=default_follow_up_mode,
            )
            for chat in list(st.session_state.chats)
        ]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = st.session_state.chats[-1]["chat_id"]

    if _find_chat_index(st.session_state.active_chat_id) is None:
        st.session_state.active_chat_id = st.session_state.chats[-1]["chat_id"]

    if "top_n" not in st.session_state:
        st.session_state.top_n = _sanitize_top_n(default_top_n)
    else:
        st.session_state.top_n = _sanitize_top_n(st.session_state.top_n)

    if "compute_device_preference" not in st.session_state:
        st.session_state.compute_device_preference = str(default_compute_device_preference or "auto")
    if "effective_compute_device" not in st.session_state:
        st.session_state.effective_compute_device = "cpu"
    if "compute_device_warning" not in st.session_state:
        st.session_state.compute_device_warning = ""
    if "last_device_applied" not in st.session_state:
        st.session_state.last_device_applied = ""
    if "edit_target" not in st.session_state:
        st.session_state.edit_target = None
    if "pending_branch_submission" not in st.session_state:
        st.session_state.pending_branch_submission = None

    _ensure_active_branch()
    _load_active_messages()
    st.session_state.session_id = st.session_state.active_chat_id


def get_active_chat_id() -> str:
    return str(st.session_state.active_chat_id)


def get_active_messages() -> list[dict]:
    return list(st.session_state.messages)


def get_active_history_messages() -> list[dict]:
    messages = list(get_active_messages())
    if messages and messages[-1].get("role") == "user":
        messages = messages[:-1]
    return messages


def append_active_message(message: dict) -> None:
    st.session_state.messages.append(dict(message))
    _sync_active_messages()


def set_active_messages(messages: list[dict]) -> None:
    st.session_state.messages = [dict(message) for message in messages]
    _sync_active_messages()


def clear_active_messages() -> None:
    st.session_state.messages = []
    current = get_active_context_state()
    st.session_state.context_state = _default_context_state(
        default_show_papers=bool(current.get("show_papers", False)),
        default_show_rewritten_query=bool(current.get("show_rewritten_query", False)),
        default_auto_scroll=bool(current.get("auto_scroll", True)),
        default_follow_up_mode=bool(current.get("follow_up_mode", True)),
    )
    _sync_active_messages()


def new_chat() -> str:
    _sync_active_messages()
    current = get_active_context_state()
    record = _new_chat_record(
        default_show_papers=bool(current.get("show_papers", False)),
        default_show_rewritten_query=bool(current.get("show_rewritten_query", False)),
        default_auto_scroll=bool(current.get("auto_scroll", True)),
        default_follow_up_mode=bool(current.get("follow_up_mode", True)),
    )
    st.session_state.chats.append(record)
    st.session_state.active_chat_id = record["chat_id"]
    st.session_state.session_id = record["chat_id"]
    _load_active_messages()
    clear_edit_target()
    clear_pending_branch_submission()
    return record["chat_id"]


def switch_chat(chat_id: str) -> None:
    if chat_id == st.session_state.active_chat_id:
        return
    _sync_active_messages()
    st.session_state.active_chat_id = chat_id
    st.session_state.session_id = chat_id
    _ensure_active_branch()
    _load_active_messages()
    clear_edit_target()
    clear_pending_branch_submission()


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


def set_show_papers(enabled: bool) -> None:
    update_active_context_state(show_papers=bool(enabled))


def get_show_papers() -> bool:
    return bool(get_active_context_state().get("show_papers", False))


def set_show_rewritten_query(enabled: bool) -> None:
    update_active_context_state(show_rewritten_query=bool(enabled))


def get_show_rewritten_query() -> bool:
    return bool(get_active_context_state().get("show_rewritten_query", False))


def set_auto_scroll(enabled: bool) -> None:
    update_active_context_state(auto_scroll=bool(enabled))


def get_auto_scroll() -> bool:
    return bool(get_active_context_state().get("auto_scroll", True))


def set_conversation_summary(summary: str) -> None:
    update_active_context_state(conversation_summary=_clip_summary(summary))


def get_conversation_summary() -> str:
    return str(get_active_context_state().get("conversation_summary", "") or "")


def update_conversation_summary(messages: list[dict] | None = None, *, max_chars: int = 320) -> str:
    rows = messages if messages is not None else get_active_messages()
    summary = summarize_messages(rows, max_chars=max_chars)
    set_conversation_summary(summary)
    return summary


def get_active_branch_id() -> str:
    chat = _get_active_chat_record()
    return str(chat.get("active_branch_id", MAIN_BRANCH_ID) or MAIN_BRANCH_ID)


def get_active_branch_record() -> dict:
    chat = _get_active_chat_record()
    branches = chat.get("branches", {}) or {}
    branch_id = str(chat.get("active_branch_id", MAIN_BRANCH_ID) or MAIN_BRANCH_ID)
    branch = branches.get(branch_id) or branches.get(MAIN_BRANCH_ID) or _new_branch_record(
        branch_id=branch_id
    )
    return dict(branch)


def get_active_chat_title() -> str:
    chat = _get_active_chat_record()
    return str(chat.get("title", "") or "New conversation")


def get_branches_for_active_chat() -> list[dict]:
    chat = _get_active_chat_record()
    branches = chat.get("branches", {}) or {}
    active_branch_id = str(chat.get("active_branch_id", MAIN_BRANCH_ID) or MAIN_BRANCH_ID)
    rows: list[dict] = []
    for branch in branches.values():
        branch_payload = dict(branch)
        branch_payload["is_active"] = str(branch_payload.get("branch_id", "")) == active_branch_id
        branch_payload["message_count"] = len(branch_payload.get("messages", []) or [])
        rows.append(branch_payload)
    return sorted(rows, key=lambda item: str(item.get("created_at", "")))


def switch_branch(branch_id: str) -> None:
    _sync_active_messages()
    chat_idx = _find_chat_index(get_active_chat_id())
    if chat_idx is None:
        return
    branches = st.session_state.chats[chat_idx].get("branches", {}) or {}
    if branch_id not in branches:
        return
    st.session_state.chats[chat_idx]["active_branch_id"] = branch_id
    _load_active_messages()
    clear_edit_target()
    clear_pending_branch_submission()


def create_branch_from_edit(message_index: int, edited_content: str) -> str:
    _sync_active_messages()
    new_messages = build_branched_messages(
        get_active_messages(),
        message_index=message_index,
        edited_content=edited_content,
    )
    current_context = get_active_context_state()
    chat_idx = _find_chat_index(get_active_chat_id())
    if chat_idx is None:
        raise ValueError("Active chat not found.")

    branch_id = f"branch-{uuid4().hex[:8]}"
    new_branch = _new_branch_record(
        branch_id=branch_id,
        messages=new_messages,
        parent_branch_id=get_active_branch_id(),
        parent_turn_index=int(message_index),
        default_show_papers=bool(current_context.get("show_papers", False)),
        default_show_rewritten_query=bool(current_context.get("show_rewritten_query", False)),
        default_auto_scroll=bool(current_context.get("auto_scroll", True)),
        default_follow_up_mode=bool(current_context.get("follow_up_mode", True)),
    )
    new_branch["context_state"]["conversation_summary"] = summarize_messages(new_messages)
    chat = st.session_state.chats[chat_idx]
    branches = chat.get("branches", {}) or {}
    branches[branch_id] = new_branch
    chat["branches"] = branches
    chat["active_branch_id"] = branch_id
    st.session_state.chats[chat_idx] = chat
    _load_active_messages()
    clear_edit_target()
    return branch_id


def build_branched_messages(
    messages: list[dict],
    *,
    message_index: int,
    edited_content: str,
) -> list[dict]:
    safe_index = int(message_index)
    if safe_index < 0 or safe_index >= len(messages):
        raise IndexError("Message index out of range.")
    target = dict(messages[safe_index] or {})
    if str(target.get("role", "") or "") != "user":
        raise ValueError("Only user messages can be branched.")

    edited = str(edited_content or "").strip()
    if not edited:
        raise ValueError("Edited content cannot be empty.")

    prefix = [dict(message) for message in messages[:safe_index]]
    updated = dict(target)
    updated["content"] = edited
    return prefix + [updated]


def set_compute_device(preference: str, effective: str, warning: str = "") -> None:
    st.session_state.compute_device_preference = str(preference or "auto")
    st.session_state.effective_compute_device = str(effective or "cpu")
    st.session_state.compute_device_warning = str(warning or "")
    st.session_state.last_device_applied = str(effective or "cpu")


def get_compute_device_preference() -> str:
    return str(st.session_state.get("compute_device_preference", "auto") or "auto")


def get_effective_compute_device() -> str:
    return str(st.session_state.get("effective_compute_device", "cpu") or "cpu")


def get_compute_device_warning() -> str:
    return str(st.session_state.get("compute_device_warning", "") or "")


def set_edit_target(message_index: int, content: str) -> None:
    st.session_state.edit_target = {
        "message_index": int(message_index),
        "content": str(content or ""),
    }


def get_edit_target() -> dict | None:
    target = st.session_state.get("edit_target")
    if not isinstance(target, dict):
        return None
    return dict(target)


def clear_edit_target() -> None:
    st.session_state.edit_target = None


def set_pending_branch_submission(*, branch_id: str, query: str) -> None:
    st.session_state.pending_branch_submission = {
        "branch_id": str(branch_id or MAIN_BRANCH_ID),
        "query": str(query or ""),
    }


def get_pending_branch_submission() -> dict | None:
    payload = st.session_state.get("pending_branch_submission")
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def clear_pending_branch_submission() -> None:
    st.session_state.pending_branch_submission = None


def summarize_messages(messages: list[dict], *, max_chars: int = 320) -> str:
    if not messages:
        return ""
    fragments: list[str] = []
    for message in messages[-4:]:
        role = str(message.get("role", "") or "")
        content = " ".join(str(message.get("content", "") or "").split()).strip()
        if not content:
            continue
        label = "assistant" if role == "assistant" else "user"
        fragments.append(f"{label}: {content}")
    summary = " | ".join(fragments).strip()
    return summary[:max_chars]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_chat_record(
    chat_id: str | None = None,
    messages: list[dict] | None = None,
    *,
    default_show_papers: bool = False,
    default_show_rewritten_query: bool = False,
    default_auto_scroll: bool = True,
    default_follow_up_mode: bool = True,
) -> dict:
    payload = [dict(message) for message in list(messages or [])]
    main_branch = _new_branch_record(
        branch_id=MAIN_BRANCH_ID,
        messages=payload,
        default_show_papers=default_show_papers,
        default_show_rewritten_query=default_show_rewritten_query,
        default_auto_scroll=default_auto_scroll,
        default_follow_up_mode=default_follow_up_mode,
    )
    main_branch["context_state"]["conversation_summary"] = summarize_messages(payload)
    return {
        "chat_id": chat_id or str(uuid4()),
        "title": _derive_chat_title(payload),
        "created_at": _utc_now_iso(),
        "active_branch_id": MAIN_BRANCH_ID,
        "branches": {MAIN_BRANCH_ID: main_branch},
        "messages": payload,
        "context_state": dict(main_branch["context_state"]),
    }


def _new_branch_record(
    *,
    branch_id: str,
    messages: list[dict] | None = None,
    parent_branch_id: str | None = None,
    parent_turn_index: int | None = None,
    default_show_papers: bool = False,
    default_show_rewritten_query: bool = False,
    default_auto_scroll: bool = True,
    default_follow_up_mode: bool = True,
) -> dict:
    payload = [dict(message) for message in list(messages or [])]
    return {
        "branch_id": branch_id,
        "title": _derive_chat_title(payload),
        "created_at": _utc_now_iso(),
        "parent_branch_id": str(parent_branch_id or ""),
        "parent_turn_index": parent_turn_index,
        "messages": payload,
        "context_state": _default_context_state(
            default_show_papers=default_show_papers,
            default_show_rewritten_query=default_show_rewritten_query,
            default_auto_scroll=default_auto_scroll,
            default_follow_up_mode=default_follow_up_mode,
        ),
    }


def _migrate_chat_record(
    chat: dict,
    *,
    default_show_papers: bool,
    default_show_rewritten_query: bool,
    default_auto_scroll: bool,
    default_follow_up_mode: bool,
) -> dict:
    payload = dict(chat or {})
    if "branches" not in payload or not isinstance(payload.get("branches"), dict):
        main_branch = _new_branch_record(
            branch_id=MAIN_BRANCH_ID,
            messages=list(payload.get("messages", []) or []),
            default_show_papers=default_show_papers,
            default_show_rewritten_query=default_show_rewritten_query,
            default_auto_scroll=default_auto_scroll,
            default_follow_up_mode=default_follow_up_mode,
        )
        legacy_context = payload.get("context_state")
        if isinstance(legacy_context, dict):
            merged_context = dict(main_branch["context_state"])
            merged_context.update(legacy_context)
            main_branch["context_state"] = merged_context
        main_branch["context_state"]["conversation_summary"] = summarize_messages(
            main_branch.get("messages", []) or []
        )
        payload["branches"] = {MAIN_BRANCH_ID: main_branch}
        payload["active_branch_id"] = MAIN_BRANCH_ID
    else:
        branches: dict[str, dict] = {}
        for branch_id, branch in payload.get("branches", {}).items():
            branch_payload = dict(branch or {})
            branch_payload.setdefault("branch_id", branch_id)
            branch_messages = [dict(message) for message in list(branch_payload.get("messages", []) or [])]
            branch_payload["messages"] = branch_messages
            context_state = branch_payload.get("context_state")
            if not isinstance(context_state, dict):
                context_state = _default_context_state(
                    default_show_papers=default_show_papers,
                    default_show_rewritten_query=default_show_rewritten_query,
                    default_auto_scroll=default_auto_scroll,
                    default_follow_up_mode=default_follow_up_mode,
                )
            else:
                merged = _default_context_state(
                    default_show_papers=default_show_papers,
                    default_show_rewritten_query=default_show_rewritten_query,
                    default_auto_scroll=default_auto_scroll,
                    default_follow_up_mode=default_follow_up_mode,
                )
                merged.update(context_state)
                context_state = merged
            context_state["conversation_summary"] = context_state.get("conversation_summary") or summarize_messages(
                branch_messages
            )
            branch_payload["context_state"] = context_state
            branch_payload.setdefault("title", _derive_chat_title(branch_messages))
            branch_payload.setdefault("created_at", _utc_now_iso())
            branch_payload.setdefault("parent_branch_id", "")
            branch_payload.setdefault("parent_turn_index", None)
            branches[str(branch_payload["branch_id"])] = branch_payload
        payload["branches"] = branches or {
            MAIN_BRANCH_ID: _new_branch_record(
                branch_id=MAIN_BRANCH_ID,
                default_show_papers=default_show_papers,
                default_show_rewritten_query=default_show_rewritten_query,
                default_auto_scroll=default_auto_scroll,
                default_follow_up_mode=default_follow_up_mode,
            )
        }

    active_branch_id = str(payload.get("active_branch_id", MAIN_BRANCH_ID) or MAIN_BRANCH_ID)
    if active_branch_id not in payload["branches"]:
        active_branch_id = next(iter(payload["branches"].keys()))
    payload["active_branch_id"] = active_branch_id
    active_branch = payload["branches"][active_branch_id]
    payload["messages"] = list(active_branch.get("messages", []) or [])
    payload["context_state"] = dict(active_branch.get("context_state", _default_context_state()))
    payload["title"] = payload.get("title") or _derive_chat_title(payload["messages"])
    payload["created_at"] = payload.get("created_at") or _utc_now_iso()
    return payload


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


def _ensure_active_branch() -> None:
    idx = _find_chat_index(st.session_state.active_chat_id)
    if idx is None:
        return
    chat = st.session_state.chats[idx]
    branches = chat.get("branches", {}) or {}
    active_branch_id = str(chat.get("active_branch_id", MAIN_BRANCH_ID) or MAIN_BRANCH_ID)
    if active_branch_id not in branches:
        active_branch_id = next(iter(branches.keys())) if branches else MAIN_BRANCH_ID
        chat["active_branch_id"] = active_branch_id
    st.session_state.chats[idx] = chat


def _load_active_messages() -> None:
    idx = _find_chat_index(st.session_state.active_chat_id)
    if idx is None:
        st.session_state.messages = []
        st.session_state.context_state = _default_context_state()
        return
    chat = st.session_state.chats[idx]
    branches = chat.get("branches", {}) or {}
    branch_id = str(chat.get("active_branch_id", MAIN_BRANCH_ID) or MAIN_BRANCH_ID)
    branch = branches.get(branch_id) or branches.get(MAIN_BRANCH_ID)
    if not isinstance(branch, dict):
        branch = _new_branch_record(branch_id=branch_id)
        branches[branch_id] = branch
        chat["branches"] = branches
    st.session_state.messages = list(branch.get("messages", []))
    context_state = branch.get("context_state")
    if not isinstance(context_state, dict):
        context_state = _default_context_state()
        branch["context_state"] = context_state
    st.session_state.context_state = dict(context_state)
    chat["messages"] = list(st.session_state.messages)
    chat["context_state"] = dict(st.session_state.context_state)
    chat["title"] = _derive_chat_title(st.session_state.messages)
    st.session_state.chats[idx] = chat


def _sync_active_messages() -> None:
    active_id = str(st.session_state.active_chat_id)
    idx = _find_chat_index(active_id)
    if idx is None:
        st.session_state.chats.append(_new_chat_record(chat_id=active_id))
        idx = len(st.session_state.chats) - 1
    chat = dict(st.session_state.chats[idx])
    branches = chat.get("branches", {}) or {}
    branch_id = str(chat.get("active_branch_id", MAIN_BRANCH_ID) or MAIN_BRANCH_ID)
    branch = dict(branches.get(branch_id) or _new_branch_record(branch_id=branch_id))
    branch["messages"] = [dict(message) for message in st.session_state.messages]
    branch["title"] = _derive_chat_title(branch["messages"])
    context_state = st.session_state.get("context_state")
    if not isinstance(context_state, dict):
        context_state = _default_context_state()
    branch["context_state"] = dict(context_state)
    branches[branch_id] = branch
    chat["branches"] = branches
    chat["messages"] = list(branch["messages"])
    chat["title"] = _derive_chat_title(branch["messages"])
    chat["context_state"] = dict(branch["context_state"])
    st.session_state.chats[idx] = chat


def _get_active_chat_record() -> dict:
    idx = _find_chat_index(get_active_chat_id())
    if idx is None:
        raise RuntimeError("Active chat not found in session state.")
    return dict(st.session_state.chats[idx])


def _sanitize_top_n(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 10
    return max(1, min(10, parsed))


def _default_context_state(
    *,
    default_show_papers: bool = False,
    default_show_rewritten_query: bool = False,
    default_auto_scroll: bool = True,
    default_follow_up_mode: bool = True,
) -> dict:
    return {
        "follow_up_mode": bool(default_follow_up_mode),
        "show_papers": bool(default_show_papers),
        "show_rewritten_query": bool(default_show_rewritten_query),
        "auto_scroll": bool(default_auto_scroll),
        "last_topic_summary": "",
        "conversation_summary": "",
        "last_retrieved_sources": [],
        "last_response_metrics": {},
        "last_answer_cache": {},
    }


def _clip_summary(summary: str, max_chars: int = 320) -> str:
    return " ".join(str(summary or "").split())[:max_chars]
