from __future__ import annotations

from typing import Any

__all__ = [
    "PaperContent",
    "PaperIndexer",
    "PaperStore",
    "extract_text_from_uploaded_pdf",
    "fetch_paper_content",
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
    raise AttributeError(name)
