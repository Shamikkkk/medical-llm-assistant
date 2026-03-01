# Created by Codex - Section 1

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceItemModel(BaseModel):
    rank: int | None = None
    pmid: str | None = None
    title: str | None = None
    journal: str | None = None
    year: str | None = None
    doi: str | None = None
    pmcid: str | None = None
    fulltext_url: str | None = None
    context: str | None = None

    model_config = ConfigDict(extra="allow")


class MessageRecord(BaseModel):
    role: str
    content: str
    created_at: str | None = None
    status: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_contexts: list[dict[str, Any]] = Field(default_factory=list)
    validation_issues: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class BranchRecord(BaseModel):
    branch_id: str
    title: str
    parent_branch_id: str = ""
    parent_turn_index: int | None = None
    created_at: str
    updated_at: str | None = None
    message_count: int = 0

    model_config = ConfigDict(extra="allow")


class SessionRecord(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str | None = None
    branch_count: int = 1

    model_config = ConfigDict(extra="allow")


class BranchCreateRequest(BaseModel):
    parent_branch_id: str
    fork_message_index: int
    edited_query: str


class ChatRequest(BaseModel):
    query: str
    session_id: str
    branch_id: str = "main"
    top_n: int = 10
    agent_mode: bool = False
    follow_up_mode: bool = True
    chat_messages: list[dict[str, Any]] = Field(default_factory=list)
    show_papers: bool = True
    conversation_summary: str = ""
    compute_device: str | None = None


class PipelineResponseModel(BaseModel):
    status: str | None = None
    answer: str | None = None
    message: str | None = None
    query: str | None = None
    request_id: str | None = None
    sources: list[SourceItemModel] = Field(default_factory=list)
    docs_preview: list[SourceItemModel] = Field(default_factory=list)
    pubmed_query: str | None = None
    reranker_active: bool | None = None
    scope_label: str | None = None
    scope_message: str | None = None
    reframed_query: str | None = None
    intent_label: str | None = None
    intent_confidence: float | None = None
    reframe_note: str | None = None
    retrieved_contexts: list[dict[str, Any]] = Field(default_factory=list)
    effective_query: str | None = None
    rewritten_query: str | None = None
    cache_hit: bool | None = None
    cache_status: str | None = None
    alignment_issues: list[str] = Field(default_factory=list)
    validation_warning: str | None = None
    validation_issues: list[str] = Field(default_factory=list)
    validation_confidence: str | None = None
    validation_suggested_fix: str | None = None
    branch_id: str | None = None
    answer_cache_hit: bool | None = None
    answer_cache_match_type: str | None = None
    answer_cache_created_at: str | None = None
    answer_cache_similarity: float | None = None
    answer_cache_query: str | None = None
    answer_cache_config_match: bool | None = None
    answer_cache_note: str | None = None
    source_count_note: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    evidence_quality: str | None = None
    invalid_citations: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class ChatResponse(BaseModel):
    payload: PipelineResponseModel
