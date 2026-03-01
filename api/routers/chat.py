# Created by Codex - Section 1

from __future__ import annotations

from typing import Any, Iterator
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_session_store
from api.models import ChatRequest, ChatResponse, MessageRecord
from api.session_store import SessionStore
from src.chat.router import invoke_chat_request, stream_chat_request

LOGGER = logging.getLogger("api.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/invoke", response_model=ChatResponse)
def invoke_chat(
    request: ChatRequest,
    session_store: SessionStore = Depends(get_session_store),
) -> ChatResponse:
    _append_user_message_if_needed(session_store, request)
    payload = invoke_chat_request(**request.model_dump())
    _append_assistant_message(session_store, request, payload)
    return ChatResponse(payload=payload)


@router.post("/stream")
def stream_chat(
    request: ChatRequest,
    session_store: SessionStore = Depends(get_session_store),
) -> StreamingResponse:
    _append_user_message_if_needed(session_store, request)

    def event_stream() -> Iterator[str]:
        stream = stream_chat_request(**request.model_dump())
        try:
            while True:
                try:
                    chunk = next(stream)
                except StopIteration as exc:
                    payload = dict(exc.value or {})
                    _append_assistant_message(session_store, request, payload)
                    yield _sse_event({"type": "done", "payload": payload})
                    break
                if chunk:
                    yield _sse_event({"type": "chunk", "text": str(chunk)})
        except Exception as exc:
            LOGGER.exception("Streaming chat request failed")
            yield _sse_event({"type": "error", "message": str(exc) or "Streaming failed."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _append_user_message_if_needed(session_store: SessionStore, request: ChatRequest) -> None:
    existing_messages = session_store.get_messages(request.session_id, request.branch_id)
    if existing_messages:
        last_message = existing_messages[-1]
        if (
            str(last_message.get("role", "") or "").strip().lower() == "user"
            and str(last_message.get("content", "") or "") == request.query
        ):
            return
    session_store.append_message(
        request.session_id,
        request.branch_id,
        MessageRecord(role="user", content=request.query).model_dump(),
    )


def _append_assistant_message(
    session_store: SessionStore,
    request: ChatRequest,
    payload: dict[str, Any],
) -> None:
    answer_text = str(payload.get("answer") or payload.get("message") or "")
    status = str(payload.get("status", "answered") or "answered")
    session_store.append_message(
        request.session_id,
        request.branch_id,
        MessageRecord(
            role="assistant",
            content=answer_text,
            status=status,
            sources=list(payload.get("sources") or []),
            retrieved_contexts=list(payload.get("retrieved_contexts") or []),
            validation_issues=list(payload.get("validation_issues") or []),
            invalid_citations=list(payload.get("invalid_citations") or []),
            timings=dict(payload.get("timings") or {}),
            answer_cache_hit=bool(payload.get("answer_cache_hit", False)),
            evidence_quality=str(payload.get("evidence_quality", "") or ""),
        ).model_dump(exclude_none=True),
    )


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
