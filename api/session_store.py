# Created by Codex - Section 1

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

MAIN_BRANCH_ID = "main"


def build_branched_messages(
    messages: list[dict[str, Any]],
    message_index: int,
    edited_content: str,
) -> list[dict[str, Any]]:
    if message_index < 0 or message_index >= len(messages):
        raise IndexError("message_index out of range")

    target = messages[message_index]
    role = str(target.get("role", "") or "").strip().lower()
    if role != "user":
        raise ValueError("Only user messages can be edited into a branch")

    edited_text = str(edited_content or "").strip()
    if not edited_text:
        raise ValueError("edited_content must not be empty")

    branched = [dict(message) for message in messages[:message_index]]
    branched.append({"role": "user", "content": edited_text})
    return branched


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._chats: dict[str, dict[str, Any]] = {}
        self._load()

    def create_chat(self, title: str) -> dict[str, Any]:
        now = _iso_now()
        chat_id = uuid4().hex
        chat_title = _normalize_title(title, fallback="New chat")
        record = {
            "chat_id": chat_id,
            "title": chat_title,
            "created_at": now,
            "updated_at": now,
            "branches": {
                MAIN_BRANCH_ID: {
                    "branch_id": MAIN_BRANCH_ID,
                    "title": "Main",
                    "parent_branch_id": "",
                    "parent_turn_index": None,
                    "created_at": now,
                    "updated_at": now,
                    "messages": [],
                }
            },
        }
        with self._lock:
            self._chats[chat_id] = record
            self._persist()
            return self._serialize_chat(record)

    def ensure_chat(self, chat_id: str, *, title: str = "New chat") -> dict[str, Any]:
        normalized_chat_id = str(chat_id or "").strip()
        if not normalized_chat_id:
            created = self.create_chat(title)
            return self._chats[str(created["chat_id"])]
        with self._lock:
            existing = self._chats.get(normalized_chat_id)
            if existing is not None:
                return existing
            now = _iso_now()
            record = {
                "chat_id": normalized_chat_id,
                "title": _normalize_title(title, fallback="New chat"),
                "created_at": now,
                "updated_at": now,
                "branches": {
                    MAIN_BRANCH_ID: {
                        "branch_id": MAIN_BRANCH_ID,
                        "title": "Main",
                        "parent_branch_id": "",
                        "parent_turn_index": None,
                        "created_at": now,
                        "updated_at": now,
                        "messages": [],
                    }
                },
            }
            self._chats[normalized_chat_id] = record
            self._persist()
            return record

    def get_chats(self) -> list[dict[str, Any]]:
        with self._lock:
            chats = [self._serialize_chat(chat) for chat in self._chats.values()]
        return sorted(
            chats,
            key=lambda item: str(item.get("updated_at", "") or item.get("created_at", "")),
            reverse=True,
        )

    def delete_chat(self, chat_id: str) -> bool:
        normalized_chat_id = str(chat_id or "").strip()
        with self._lock:
            existed = normalized_chat_id in self._chats
            if existed:
                self._chats.pop(normalized_chat_id, None)
                self._persist()
            return existed

    def get_branches(self, chat_id: str) -> list[dict[str, Any]]:
        with self._lock:
            chat = self._chats.get(str(chat_id or "").strip())
            if chat is None:
                return []
            branches = [
                self._serialize_branch(branch)
                for branch in chat.get("branches", {}).values()
            ]
        return sorted(branches, key=lambda item: str(item.get("created_at", "")))

    def ensure_branch(self, chat_id: str, branch_id: str) -> dict[str, Any]:
        chat = self.ensure_chat(chat_id)
        normalized_branch_id = str(branch_id or "").strip() or MAIN_BRANCH_ID
        with self._lock:
            branch = chat["branches"].get(normalized_branch_id)
            if branch is not None:
                return branch
            now = _iso_now()
            branch = {
                "branch_id": normalized_branch_id,
                "title": normalized_branch_id,
                "parent_branch_id": "",
                "parent_turn_index": None,
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            chat["branches"][normalized_branch_id] = branch
            chat["updated_at"] = now
            self._persist()
            return branch

    def create_branch(
        self,
        chat_id: str,
        parent_branch_id: str,
        fork_message_index: int,
        edited_query: str,
    ) -> dict[str, Any]:
        normalized_chat_id = str(chat_id or "").strip()
        normalized_parent_branch_id = str(parent_branch_id or "").strip() or MAIN_BRANCH_ID
        with self._lock:
            chat = self._chats.get(normalized_chat_id)
            if chat is None:
                raise KeyError(f"Unknown chat_id: {normalized_chat_id}")
            parent_branch = chat.get("branches", {}).get(normalized_parent_branch_id)
            if parent_branch is None:
                raise KeyError(f"Unknown branch_id: {normalized_parent_branch_id}")
            branched_messages = build_branched_messages(
                list(parent_branch.get("messages", [])),
                int(fork_message_index),
                edited_query,
            )
            now = _iso_now()
            branch_id = uuid4().hex
            branch = {
                "branch_id": branch_id,
                "title": _normalize_title(edited_query, fallback="Branched thread"),
                "parent_branch_id": normalized_parent_branch_id,
                "parent_turn_index": int(fork_message_index),
                "created_at": now,
                "updated_at": now,
                "messages": [
                    _normalize_message(message)
                    for message in branched_messages
                ],
            }
            chat["branches"][branch_id] = branch
            chat["updated_at"] = now
            self._persist()
            return self._serialize_branch(branch)

    def get_messages(self, chat_id: str, branch_id: str) -> list[dict[str, Any]]:
        with self._lock:
            chat = self._chats.get(str(chat_id or "").strip())
            if chat is None:
                return []
            branch = chat.get("branches", {}).get(str(branch_id or "").strip() or MAIN_BRANCH_ID)
            if branch is None:
                return []
            return [dict(message) for message in branch.get("messages", [])]

    def append_message(self, chat_id: str, branch_id: str, message: dict[str, Any]) -> None:
        chat = self.ensure_chat(chat_id, title=str(message.get("content", "") or "New chat"))
        branch = self.ensure_branch(chat_id, branch_id)
        normalized_message = _normalize_message(message)
        now = normalized_message["created_at"]
        with self._lock:
            branch["messages"].append(normalized_message)
            branch["updated_at"] = now
            if len(branch["messages"]) == 1 and str(branch.get("title", "") or "").strip() == MAIN_BRANCH_ID:
                branch["title"] = "Main"
            if not str(chat.get("title", "") or "").strip() or str(chat.get("title")) == "New chat":
                if normalized_message["role"] == "user":
                    chat["title"] = _normalize_title(normalized_message["content"], fallback="New chat")
            chat["updated_at"] = now
            self._persist()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        chats = payload.get("chats", {})
        if not isinstance(chats, dict):
            return
        loaded: dict[str, dict[str, Any]] = {}
        for chat_id, raw_chat in chats.items():
            if not isinstance(raw_chat, dict):
                continue
            normalized_chat_id = str(chat_id or raw_chat.get("chat_id", "") or "").strip()
            if not normalized_chat_id:
                continue
            loaded[normalized_chat_id] = self._normalize_chat(raw_chat, normalized_chat_id)
        self._chats = loaded

    def _persist(self) -> None:
        payload = {
            "chats": self._chats,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _normalize_chat(self, raw_chat: dict[str, Any], chat_id: str) -> dict[str, Any]:
        created_at = str(raw_chat.get("created_at", "") or _iso_now())
        updated_at = str(raw_chat.get("updated_at", "") or created_at)
        raw_branches = raw_chat.get("branches", {})
        branches: dict[str, dict[str, Any]] = {}
        if isinstance(raw_branches, dict):
            for branch_id, raw_branch in raw_branches.items():
                if not isinstance(raw_branch, dict):
                    continue
                normalized_branch_id = (
                    str(branch_id or raw_branch.get("branch_id", "") or "").strip()
                    or MAIN_BRANCH_ID
                )
                branches[normalized_branch_id] = self._normalize_branch(
                    raw_branch,
                    normalized_branch_id,
                )
        if MAIN_BRANCH_ID not in branches:
            branches[MAIN_BRANCH_ID] = {
                "branch_id": MAIN_BRANCH_ID,
                "title": "Main",
                "parent_branch_id": "",
                "parent_turn_index": None,
                "created_at": created_at,
                "updated_at": updated_at,
                "messages": [],
            }
        return {
            "chat_id": chat_id,
            "title": _normalize_title(raw_chat.get("title"), fallback="New chat"),
            "created_at": created_at,
            "updated_at": updated_at,
            "branches": branches,
        }

    def _normalize_branch(self, raw_branch: dict[str, Any], branch_id: str) -> dict[str, Any]:
        created_at = str(raw_branch.get("created_at", "") or _iso_now())
        updated_at = str(raw_branch.get("updated_at", "") or created_at)
        raw_messages = raw_branch.get("messages", [])
        messages = [
            _normalize_message(message)
            for message in raw_messages
            if isinstance(message, dict)
        ]
        return {
            "branch_id": branch_id,
            "title": _normalize_title(
                raw_branch.get("title"),
                fallback="Main" if branch_id == MAIN_BRANCH_ID else branch_id,
            ),
            "parent_branch_id": str(raw_branch.get("parent_branch_id", "") or ""),
            "parent_turn_index": raw_branch.get("parent_turn_index"),
            "created_at": created_at,
            "updated_at": updated_at,
            "messages": messages,
        }

    def _serialize_chat(self, chat: dict[str, Any]) -> dict[str, Any]:
        branches = chat.get("branches", {})
        return {
            "chat_id": chat["chat_id"],
            "title": chat["title"],
            "created_at": chat["created_at"],
            "updated_at": chat.get("updated_at"),
            "branch_count": len(branches),
        }

    def _serialize_branch(self, branch: dict[str, Any]) -> dict[str, Any]:
        return {
            "branch_id": branch["branch_id"],
            "title": branch["title"],
            "parent_branch_id": branch.get("parent_branch_id", ""),
            "parent_turn_index": branch.get("parent_turn_index"),
            "created_at": branch["created_at"],
            "updated_at": branch.get("updated_at"),
            "message_count": len(branch.get("messages", [])),
        }


def _normalize_title(value: Any, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    compact = " ".join(text.split())
    if len(compact) <= 80:
        return compact
    return compact[:77].rstrip() + "..."


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(message)
    normalized["role"] = str(normalized.get("role", "") or "").strip() or "user"
    normalized["content"] = str(normalized.get("content", "") or "")
    normalized["created_at"] = str(normalized.get("created_at", "") or _iso_now())
    if "sources" not in normalized or not isinstance(normalized.get("sources"), list):
        normalized["sources"] = []
    if "retrieved_contexts" not in normalized or not isinstance(
        normalized.get("retrieved_contexts"), list
    ):
        normalized["retrieved_contexts"] = []
    if "validation_issues" not in normalized or not isinstance(
        normalized.get("validation_issues"), list
    ):
        normalized["validation_issues"] = []
    if "invalid_citations" not in normalized or not isinstance(
        normalized.get("invalid_citations"), list
    ):
        normalized["invalid_citations"] = []
    if "timings" not in normalized or not isinstance(normalized.get("timings"), dict):
        normalized["timings"] = {}
    return normalized


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
