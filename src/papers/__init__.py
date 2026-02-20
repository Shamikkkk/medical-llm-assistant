from __future__ import annotations

from typing import Any

__all__ = [
    "PaperContent",
    "PaperIndexer",
    "PaperStore",
    "FullTextDiscoveryResult",
    "discover_fulltext",
    "extract_text_from_uploaded_pdf",
    "fetch_readable_text_from_url",
    "fetch_paper_content",
    "STATUS_OK_OA_HTML",
    "STATUS_PAYWALLED_OR_BLOCKED",
    "STATUS_UNSUPPORTED_CONTENT",
    "STATUS_ERROR",
]


def __getattr__(name: str) -> Any:
    if name in {"PaperContent", "PaperStore"}:
        from src.papers.store import PaperContent, PaperStore

        return {"PaperContent": PaperContent, "PaperStore": PaperStore}[name]
    if name == "PaperIndexer":
        from src.papers.index import PaperIndexer

        return PaperIndexer
    if name in {
        "extract_text_from_uploaded_pdf",
        "fetch_paper_content",
    }:
        from src.papers.fetch import extract_text_from_uploaded_pdf, fetch_paper_content

        return {
            "extract_text_from_uploaded_pdf": extract_text_from_uploaded_pdf,
            "fetch_paper_content": fetch_paper_content,
        }[name]
    if name in {
        "STATUS_OK_OA_HTML",
        "STATUS_PAYWALLED_OR_BLOCKED",
        "STATUS_UNSUPPORTED_CONTENT",
        "STATUS_ERROR",
        "fetch_readable_text_from_url",
    }:
        from src.papers.fetch_fulltext import (
            STATUS_ERROR,
            STATUS_OK_OA_HTML,
            STATUS_PAYWALLED_OR_BLOCKED,
            STATUS_UNSUPPORTED_CONTENT,
            fetch_readable_text_from_url,
        )

        return {
            "STATUS_OK_OA_HTML": STATUS_OK_OA_HTML,
            "STATUS_PAYWALLED_OR_BLOCKED": STATUS_PAYWALLED_OR_BLOCKED,
            "STATUS_UNSUPPORTED_CONTENT": STATUS_UNSUPPORTED_CONTENT,
            "STATUS_ERROR": STATUS_ERROR,
            "fetch_readable_text_from_url": fetch_readable_text_from_url,
        }[name]
    if name in {"FullTextDiscoveryResult", "discover_fulltext"}:
        from src.papers.fulltext_discovery import FullTextDiscoveryResult, discover_fulltext

        return {
            "FullTextDiscoveryResult": FullTextDiscoveryResult,
            "discover_fulltext": discover_fulltext,
        }[name]
    raise AttributeError(name)
