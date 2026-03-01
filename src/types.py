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
    context: str


class PipelineResponse(TypedDict, total=False):
    status: str
    answer: str
    message: str
    query: str
    request_id: str
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
    cache_hit: bool
    cache_status: str
    alignment_issues: list[str]
    validation_warning: str
    validation_issues: list[str]
    validation_confidence: str
    validation_suggested_fix: str
    branch_id: str
    answer_cache_hit: bool
    answer_cache_match_type: str
    answer_cache_created_at: str
    answer_cache_similarity: float
    answer_cache_query: str
    answer_cache_config_match: bool
    answer_cache_note: str
    source_count_note: str
    timings: dict[str, float]
