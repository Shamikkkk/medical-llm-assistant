from __future__ import annotations

def invoke_chat_with_mode(
    query: str,
    *,
    session_id: str,
    top_n: int,
    agent_mode: bool,
):
    if agent_mode:
        return _invoke_agent(query, session_id=session_id, top_n=top_n)
    return _invoke_pipeline(query, session_id=session_id, top_n=top_n)


def stream_chat_with_mode(
    query: str,
    *,
    session_id: str,
    top_n: int,
    agent_mode: bool,
):
    if agent_mode:
        return _stream_agent(query, session_id=session_id, top_n=top_n)
    return _stream_pipeline(query, session_id=session_id, top_n=top_n)


def _invoke_agent(query: str, *, session_id: str, top_n: int):
    from src.agent.orchestrator import invoke_agent_chat

    return invoke_agent_chat(query, session_id=session_id, top_n=top_n)


def _invoke_pipeline(query: str, *, session_id: str, top_n: int):
    from src.core.pipeline import invoke_chat

    return invoke_chat(query, session_id=session_id, top_n=top_n)


def _stream_agent(query: str, *, session_id: str, top_n: int):
    from src.agent.orchestrator import stream_agent_chat

    return stream_agent_chat(query, session_id=session_id, top_n=top_n)


def _stream_pipeline(query: str, *, session_id: str, top_n: int):
    from src.core.pipeline import stream_chat

    return stream_chat(query, session_id=session_id, top_n=top_n)
