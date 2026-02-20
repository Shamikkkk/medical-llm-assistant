from __future__ import annotations

import logging
import re
from typing import Any

LOGGER = logging.getLogger("chat.contextualize")

FOLLOWUP_HINT_RE = re.compile(
    r"\b(it|that|this|they|those|these|more|details|what about|follow[- ]?up)\b",
    flags=re.IGNORECASE,
)


def contextualize_question(
    *,
    user_query: str,
    chat_messages: list[dict[str, Any]],
    follow_up_mode: bool,
    llm: Any | None = None,
) -> tuple[str, str, bool]:
    """Return (effective_query, topic_summary, rewritten)."""
    query = str(user_query or "").strip()
    topic_summary = summarize_last_topic(chat_messages)
    if not follow_up_mode or not query:
        return query, topic_summary, False

    if not _needs_contextualization(query):
        return query, topic_summary, False

    history_excerpt = _history_excerpt(chat_messages)
    if not history_excerpt:
        return query, topic_summary, False

    if llm is not None:
        rewritten = _rewrite_with_llm(query=query, history=history_excerpt, llm=llm)
        if rewritten:
            LOGGER.info("[FOLLOWUP] rewritten_query='%s'", _trim(rewritten))
            return rewritten, topic_summary, rewritten != query

    # Heuristic fallback.
    if topic_summary:
        rewritten = f"{query} (context: {topic_summary})"
    else:
        rewritten = f"{query} (context: {history_excerpt[:180]})"
    LOGGER.info("[FOLLOWUP] rewritten_query='%s' (heuristic)", _trim(rewritten))
    return rewritten, topic_summary, rewritten != query


def summarize_last_topic(messages: list[dict[str, Any]], max_chars: int = 220) -> str:
    for message in reversed(messages or []):
        if str(message.get("role", "")) != "assistant":
            continue
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        summary = " ".join(content.split())
        return summary[:max_chars]
    for message in reversed(messages or []):
        if str(message.get("role", "")) != "user":
            continue
        content = str(message.get("content", "") or "").strip()
        if content:
            return " ".join(content.split())[:max_chars]
    return ""


def _needs_contextualization(query: str) -> bool:
    tokens = query.split()
    if len(tokens) <= 7:
        return True
    return bool(FOLLOWUP_HINT_RE.search(query))


def _history_excerpt(messages: list[dict[str, Any]], max_turns: int = 6, max_chars: int = 900) -> str:
    rows: list[str] = []
    for message in (messages or [])[-max_turns:]:
        role = str(message.get("role", ""))
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        speaker = "assistant" if role == "assistant" else "user"
        rows.append(f"{speaker}: {content}")
    text = "\n".join(rows).strip()
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _rewrite_with_llm(*, query: str, history: str, llm: Any) -> str | None:
    prompt = (
        "Rewrite the follow-up user question into a standalone research query while preserving intent.\n"
        "Use conversation history only to resolve references.\n"
        "Return ONLY the rewritten question.\n"
        f"History:\n{history}\n\n"
        f"User question:\n{query}\n"
    )
    try:
        if hasattr(llm, "invoke"):
            response = llm.invoke(prompt)
            text = _extract_text(response)
        elif hasattr(llm, "predict"):
            response = llm.predict(prompt)
            text = _extract_text(response)
        else:
            return None
    except Exception:
        return None

    cleaned = str(text or "").strip().strip('"').strip("'")
    if "\n" in cleaned:
        cleaned = cleaned.splitlines()[0].strip()
    if not cleaned:
        return None
    return cleaned[:320]


def _extract_text(payload: Any) -> str:
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text")
                if value:
                    parts.append(str(value))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(payload or "")


def _trim(text: str, limit: int = 140) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."
