from __future__ import annotations

from typing import TypedDict


class SourceItem(TypedDict, total=False):
    rank: int
    pmid: str
    title: str
    journal: str
    year: str
    doi: str
    pmcid: str
    fulltext_url: str


class PipelineResponse(TypedDict, total=False):
    status: str
    answer: str
    message: str
    query: str
    sources: list[SourceItem]
    docs_preview: list[SourceItem]
    pubmed_query: str
    reranker_active: bool
    scope_label: str
    scope_message: str
    reframed_query: str
    intent_label: str
    intent_confidence: float
    reframe_note: str
    retrieved_contexts: list[dict[str, str]]
    effective_query: str
    rewritten_query: str
    validation_warning: str
    validation_issues: list[str]
    validation_confidence: str
    validation_suggested_fix: str
