from __future__ import annotations

def invoke_chat_with_mode(
    query: str,
    *,
    session_id: str,
    branch_id: str = "main",
    top_n: int,
    agent_mode: bool,
    request_id: str | None = None,
    include_paper_links: bool = True,
    compute_device: str | None = None,
):
    if agent_mode:
        return _invoke_agent(
            query,
            session_id=session_id,
            branch_id=branch_id,
            top_n=top_n,
            request_id=request_id,
            include_paper_links=include_paper_links,
            compute_device=compute_device,
        )
    return _invoke_pipeline(
        query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=top_n,
        request_id=request_id,
        include_paper_links=include_paper_links,
        compute_device=compute_device,
    )


def stream_chat_with_mode(
    query: str,
    *,
    session_id: str,
    branch_id: str = "main",
    top_n: int,
    agent_mode: bool,
    request_id: str | None = None,
    include_paper_links: bool = True,
    compute_device: str | None = None,
):
    if agent_mode:
        return _stream_agent(
            query,
            session_id=session_id,
            branch_id=branch_id,
            top_n=top_n,
            request_id=request_id,
            include_paper_links=include_paper_links,
            compute_device=compute_device,
        )
    return _stream_pipeline(
        query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=top_n,
        request_id=request_id,
        include_paper_links=include_paper_links,
        compute_device=compute_device,
    )


def _invoke_agent(
    query: str,
    *,
    session_id: str,
    branch_id: str,
    top_n: int,
    request_id: str | None,
    include_paper_links: bool,
    compute_device: str | None,
):
    from src.agent.orchestrator import invoke_agent_chat

    return invoke_agent_chat(
        query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=top_n,
        request_id=request_id,
        include_paper_links=include_paper_links,
        compute_device=compute_device,
    )


def _invoke_pipeline(
    query: str,
    *,
    session_id: str,
    branch_id: str,
    top_n: int,
    request_id: str | None,
    include_paper_links: bool,
    compute_device: str | None,
):
    from src.core.pipeline import invoke_chat

    return invoke_chat(
        query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=top_n,
        request_id=request_id,
        include_paper_links=include_paper_links,
        compute_device=compute_device,
    )


def _stream_agent(
    query: str,
    *,
    session_id: str,
    branch_id: str,
    top_n: int,
    request_id: str | None,
    include_paper_links: bool,
    compute_device: str | None,
):
    from src.agent.orchestrator import stream_agent_chat

    return stream_agent_chat(
        query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=top_n,
        request_id=request_id,
        include_paper_links=include_paper_links,
        compute_device=compute_device,
    )


def _stream_pipeline(
    query: str,
    *,
    session_id: str,
    branch_id: str,
    top_n: int,
    request_id: str | None,
    include_paper_links: bool,
    compute_device: str | None,
):
    from src.core.pipeline import stream_chat

    return stream_chat(
        query,
        session_id=session_id,
        branch_id=branch_id,
        top_n=top_n,
        request_id=request_id,
        include_paper_links=include_paper_links,
        compute_device=compute_device,
    )
