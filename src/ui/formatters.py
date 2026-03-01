from __future__ import annotations

import json
import re


def strip_reframe_block(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Remove a leading "Reframe:" block up to first blank line or "Direct answer:".
    cleaned = re.sub(
        r"^\s*reframe:\s*.*?(?=\n\s*\n|\n\s*direct answer\s*:|$)\n?",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Remove standalone "Reframe:" lines anywhere.
    cleaned = re.sub(r"(?im)^\s*reframe:\s*.*\n?", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def beautify_text(text: str) -> str:
    if text is None:
        return ""
    cleaned = strip_reframe_block(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return ""

    # Keep citation clusters readable, e.g. "...[123][456]" -> "... [123][456]"
    cleaned = re.sub(r"(?<=\S)(\[\d)", r" \1", cleaned)

    lines = cleaned.split("\n")
    non_empty = [line for line in lines if line.strip()]
    avg_len = (
        sum(len(line) for line in non_empty) / len(non_empty)
        if non_empty
        else len(cleaned)
    )

    if "\n" not in cleaned or avg_len > 180:
        cleaned = re.sub(r"([.!?])\s+", r"\1\n\n", cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def pubmed_url(pmid: str) -> str:
    normalized = str(pmid or "").strip()
    return f"https://pubmed.ncbi.nlm.nih.gov/{normalized}/"


def doi_url(doi: str) -> str:
    normalized = str(doi or "").strip()
    if not normalized:
        return ""
    return f"https://doi.org/{normalized}"


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
