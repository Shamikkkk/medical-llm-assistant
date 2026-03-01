# Created by Codex - Section 1

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_session_store
from api.models import BranchCreateRequest
from api.session_store import MAIN_BRANCH_ID, SessionStore

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions(session_store: SessionStore = Depends(get_session_store)) -> list[dict]:
    return session_store.get_chats()


@router.post("")
def create_session(session_store: SessionStore = Depends(get_session_store)) -> dict:
    chat = session_store.create_chat("New chat")
    return {"chat_id": chat["chat_id"], "branch_id": MAIN_BRANCH_ID}


@router.delete("/{chat_id}")
def delete_session(chat_id: str, session_store: SessionStore = Depends(get_session_store)) -> dict:
    deleted = session_store.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"deleted": True}


@router.get("/{chat_id}/branches")
def list_branches(chat_id: str, session_store: SessionStore = Depends(get_session_store)) -> list[dict]:
    branches = session_store.get_branches(chat_id)
    if not branches:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return branches


@router.post("/{chat_id}/branches")
def create_branch(
    chat_id: str,
    request: BranchCreateRequest,
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    try:
        branch = session_store.create_branch(
            chat_id,
            request.parent_branch_id,
            request.fork_message_index,
            request.edited_query,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"branch_id": branch["branch_id"]}


@router.get("/{chat_id}/branches/{branch_id}/messages")
def list_branch_messages(
    chat_id: str,
    branch_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> list[dict]:
    messages = session_store.get_messages(chat_id, branch_id)
    if not messages and not session_store.get_branches(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    if not messages and branch_id not in {item["branch_id"] for item in session_store.get_branches(chat_id)}:
        raise HTTPException(status_code=404, detail="Branch not found.")
    return messages
