from __future__ import annotations

from typing import Any, TypedDict

from src.types import SourceItem


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    top_n: int
    status: str
    intent_label: str
    intent_confidence: float
    scope_label: str
    scope_message: str
    reframe_note: str
    pubmed_query: str
    retrieval_query: str
    pmids: list[str]
    records: list[dict[str, Any]]
    docs_preview: list[SourceItem]
    retrieved_contexts: list[dict[str, str]]
    sources: list[SourceItem]
    reranker_active: bool
    answer: str
    message: str
    llm_available: bool
    llm: Any
    persist_dir: str
    use_reranker: bool
    log_pipeline: bool
    scope: Any
    retriever: Any
    abstract_store: Any
