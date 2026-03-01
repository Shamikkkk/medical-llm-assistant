# Created by Codex - Section 1

from __future__ import annotations

import json


def export_branch_markdown(
    *,
    chat_title: str,
    branch_title: str,
    branch_id: str,
    parent_branch_id: str,
    messages: list[dict],
) -> str:
    lines = [
        f"# {str(chat_title or 'Medical LLM Assistant Chat').strip()}",
        "",
        f"- Branch: `{str(branch_id or 'main').strip()}`",
        f"- Branch title: {str(branch_title or 'Conversation branch').strip()}",
    ]
    if str(parent_branch_id or "").strip():
        lines.append(f"- Parent branch: `{str(parent_branch_id).strip()}`")
    lines.append("")

    for message in messages:
        role = "Assistant" if str(message.get("role", "") or "") == "assistant" else "User"
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        lines.extend([f"## {role}", "", content, ""])
    return "\n".join(lines).strip() + "\n"


def export_branch_json(
    *,
    chat_id: str,
    chat_title: str,
    branch: dict,
    messages: list[dict],
) -> str:
    payload = {
        "chat_id": str(chat_id or ""),
        "chat_title": str(chat_title or ""),
        "branch": {
            "branch_id": str(branch.get("branch_id", "") or ""),
            "title": str(branch.get("title", "") or ""),
            "parent_branch_id": str(branch.get("parent_branch_id", "") or ""),
            "parent_turn_index": branch.get("parent_turn_index"),
            "created_at": str(branch.get("created_at", "") or ""),
        },
        "messages": [dict(message) for message in messages],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
