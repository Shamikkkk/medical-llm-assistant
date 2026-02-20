from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import logging
import re
from typing import Any

import requests

LOGGER = logging.getLogger("papers.fetch_fulltext")

STATUS_OK_OA_HTML = "OK_OA_HTML"
STATUS_PAYWALLED_OR_BLOCKED = "PAYWALLED_OR_BLOCKED"
STATUS_UNSUPPORTED_CONTENT = "UNSUPPORTED_CONTENT"
STATUS_ERROR = "ERROR"

FETCH_TIMEOUT_SECONDS = 20
USER_AGENT = (
    "medical-llm-assistant/1.0 "
    "(academic use; no paywall bypass; +https://pubmed.ncbi.nlm.nih.gov)"
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

PAYWALL_MARKERS = (
    "purchase this article",
    "subscribe to continue",
    "institutional login",
    "sign in to access",
    "buy this article",
    "access through your institution",
    "rent this article",
    "this is a preview",
    "full text options",
)


def fetch_readable_text_from_url(url: str) -> tuple[str, dict[str, Any], str]:
    normalized = str(url or "").strip()
    if not normalized:
        return "", {"url": normalized, "reason": "empty_url"}, STATUS_ERROR

    try:
        response = requests.get(
            normalized,
            timeout=FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
            headers=REQUEST_HEADERS,
        )
    except requests.RequestException as exc:
        return "", {"url": normalized, "reason": str(exc)}, STATUS_ERROR

    final_url = str(response.url or normalized)
    content_type = str(response.headers.get("Content-Type", "") or "").lower()
    meta: dict[str, Any] = {
        "url": normalized,
        "final_url": final_url,
        "status_code": int(response.status_code),
        "content_type": content_type,
    }

    if response.status_code in {401, 402, 403, 429}:
        return "", meta, STATUS_PAYWALLED_OR_BLOCKED
    if response.status_code >= 400:
        return "", meta, STATUS_ERROR

    if "html" not in content_type and "xml" not in content_type:
        return "", meta, STATUS_UNSUPPORTED_CONTENT

    html = response.text or ""
    lowered = html.lower()
    if _looks_paywalled(lowered):
        return "", meta, STATUS_PAYWALLED_OR_BLOCKED

    text = _extract_readable_text(html)
    text = _normalize_text(text)
    if len(text) < 400:
        if _looks_paywalled(lowered):
            return "", meta, STATUS_PAYWALLED_OR_BLOCKED
        return "", meta, STATUS_UNSUPPORTED_CONTENT

    meta["title"] = _extract_title(html)
    meta["char_count"] = len(text)
    return text, meta, STATUS_OK_OA_HTML


def _extract_readable_text(html: str) -> str:
    extracted = ""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        ) or ""
    except Exception:
        extracted = ""

    if extracted.strip():
        return extracted

    parser = _PlainTextHTMLParser()
    try:
        parser.feed(html)
    except Exception:
        return ""
    return parser.get_text()


def _looks_paywalled(lowered_html: str) -> bool:
    return any(marker in lowered_html for marker in PAYWALL_MARKERS)


def _normalize_text(text: str) -> str:
    normalized = unescape(str(text or ""))
    normalized = re.sub(r"\r\n?", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _normalize_text(match.group(1))


class _PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in {"p", "div", "section", "article", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = str(data or "").strip()
        if text:
            self._parts.append(text)
            self._parts.append(" ")

    def get_text(self) -> str:
        return "".join(self._parts)
