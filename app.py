from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src import app_state
from eval.dashboard import render_evaluation_dashboard
from eval.evaluator import evaluate_turn, should_sample_query
from eval.store import EvalStore
from src.chat.router import invoke_chat_request, stream_chat_request
from src.core.config import load_config
from src.history import clear_session_history
from src.logging_utils import setup_logging
from src.ui.formatters import beautify_text, strip_reframe_block
from src.ui.loading_messages import detect_topic, pick_loading_message
from src.ui.render import (
    auto_scroll,
    apply_app_styles,
    render_chat,
    render_header,
    render_ranked_sources,
    render_sidebar,
)


def _safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - legacy streamlit
        st.experimental_rerun()


def _scroll_to_bottom() -> None:
    auto_scroll()


def _consume_stream_response(stream, placeholder) -> tuple[str, dict]:
    answer_text = ""
    final_payload: dict = {}
    chunk_count = 0
    while True:
        try:
            chunk = next(stream)
        except StopIteration as exc:
            final_payload = exc.value or {}
            break
        if not chunk:
            continue
        answer_text += str(chunk)
        display_text = beautify_text(strip_reframe_block(answer_text))
        placeholder.markdown(display_text, unsafe_allow_html=False)
        chunk_count += 1
        if chunk_count % 2 == 0:
            _scroll_to_bottom()
    return answer_text, final_payload


def _merge_source_metadata(primary: list[dict], fallback: list[dict]) -> list[dict]:
    if not isinstance(primary, list):
        primary = []
    if not isinstance(fallback, list):
        fallback = []
    by_pmid: dict[str, dict] = {}
    for item in fallback:
        if not isinstance(item, dict):
            continue
        pmid = str(item.get("pmid", "") or "").strip()
        if pmid:
            by_pmid[pmid] = dict(item)

    merged: list[dict] = []
    for item in primary:
        if not isinstance(item, dict):
            continue
        pmid = str(item.get("pmid", "") or "").strip()
        if not pmid:
            merged.append(dict(item))
            continue
        base = dict(by_pmid.get(pmid, {}))
        base.update(item)
        merged.append(base)

    if merged:
        return merged
    return [dict(item) for item in fallback if isinstance(item, dict)]


def _maybe_run_online_evaluation(
    *,
    config,
    query: str,
    payload: dict,
) -> None:
    if not bool(getattr(config, "eval_mode", False)):
        return
    if str(payload.get("status", "")) != "answered":
        return
    if not should_sample_query(query, float(getattr(config, "eval_sample_rate", 0.0))):
        return
    answer = str(payload.get("answer") or payload.get("message") or "")
    contexts = payload.get("retrieved_contexts", []) or []
    sources = payload.get("sources", []) or []
    record = evaluate_turn(
        query=query,
        answer=answer,
        contexts=contexts,
        sources=sources,
        mode="online",
    )
    EvalStore(getattr(config, "eval_store_path")).append(record)


def _render_controls() -> tuple[bool, bool]:
    follow_up_mode = app_state.get_follow_up_mode()
    show_papers = app_state.get_show_papers()
    col1, col2 = st.columns(2)
    with col1:
        next_follow_up = st.toggle(
            "Follow-up mode",
            value=follow_up_mode,
            help="Use chat context to rewrite follow-up questions before retrieval.",
        )
    with col2:
        next_show_papers = st.toggle(
            "Show papers",
            value=show_papers,
            help="Display ranked paper links (PubMed + DOI) in the UI.",
        )
    if next_follow_up != follow_up_mode:
        app_state.set_follow_up_mode(next_follow_up)
    if next_show_papers != show_papers:
        app_state.set_show_papers(next_show_papers)
    return bool(next_follow_up), bool(next_show_papers)


def _render_chat_experience(config) -> None:
    top_n = app_state.get_top_n()
    follow_up_mode, show_papers = _render_controls()
    render_chat(app_state.get_active_messages(), top_n=top_n, show_papers=show_papers)
    st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)

    user_input = st.chat_input(
        "Ask a medical or health question (e.g., oncology, gut health, neurology)"
    )
    if not user_input:
        return

    app_state.append_active_message({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input, unsafe_allow_html=False)
    _scroll_to_bottom()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        loading_message = pick_loading_message(detect_topic(user_input), user_input)
        placeholder.markdown(loading_message, unsafe_allow_html=False)

        answer_text = ""
        final_payload: dict = {}
        session_id = app_state.get_active_chat_id()
        chat_messages = app_state.get_active_messages()[:-1]

        try:
            stream = stream_chat_request(
                query=user_input,
                session_id=session_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                chat_messages=chat_messages,
            )
            answer_text, final_payload = _consume_stream_response(stream, placeholder)
        except Exception:
            final_payload = invoke_chat_request(
                query=user_input,
                session_id=session_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                chat_messages=chat_messages,
            )
            answer_text = str(
                final_payload.get("answer")
                or final_payload.get("message")
                or "No answer available."
            )
            answer_text = strip_reframe_block(answer_text)
            placeholder.markdown(beautify_text(answer_text), unsafe_allow_html=False)

        if not answer_text and (final_payload.get("answer") or final_payload.get("message")):
            answer_text = str(
                final_payload.get("answer")
                or final_payload.get("message")
                or "No answer available."
            )

        answer_text = strip_reframe_block(answer_text)
        final_payload["answer"] = answer_text
        placeholder.markdown(beautify_text(answer_text), unsafe_allow_html=False)
        _scroll_to_bottom()

        rewritten_query = str(final_payload.get("rewritten_query", "") or "").strip()
        if rewritten_query:
            st.caption(f"Follow-up rewrite: {rewritten_query}")

        status = str(final_payload.get("status", "answered"))
        assistant_message = {
            "role": "assistant",
            "content": answer_text,
            "sources": final_payload.get("sources", []) or [],
            "pubmed_query": final_payload.get("pubmed_query", ""),
            "reranker_active": final_payload.get("reranker_active", False),
            "status": status,
            "validation_warning": final_payload.get("validation_warning", ""),
            "validation_issues": final_payload.get("validation_issues", []) or [],
            "rewritten_query": rewritten_query,
        }
        if status in {"out_of_scope", "smalltalk"}:
            assistant_message = {"role": "assistant", "content": answer_text, "status": status}
        else:
            warning = str(final_payload.get("validation_warning", "") or "").strip()
            issues = final_payload.get("validation_issues", []) or []
            if warning:
                st.warning(warning)
            if issues:
                with st.expander("Validation details", expanded=False):
                    for issue in issues:
                        st.markdown(f"- {issue}", unsafe_allow_html=False)

        app_state.append_active_message(assistant_message)
        existing_context = app_state.get_active_context_state()
        existing_sources = existing_context.get("last_retrieved_sources", []) or []
        payload_sources = final_payload.get("sources", []) or []
        preview_sources = final_payload.get("docs_preview", []) or []
        new_sources = _merge_source_metadata(payload_sources, preview_sources)
        if not isinstance(new_sources, list) or not new_sources:
            new_sources = existing_sources
        app_state.update_active_context_state(
            last_topic_summary=str(final_payload.get("last_topic_summary", "") or ""),
            last_retrieved_sources=new_sources,
        )
        _maybe_run_online_evaluation(config=config, query=user_input, payload=final_payload)

        if show_papers and status == "answered":
            render_ranked_sources(assistant_message.get("sources", []) or [], top_n=top_n)
        _scroll_to_bottom()


def main() -> None:
    load_dotenv(override=False)
    config = load_config()
    setup_logging(config.log_level)
    st.set_page_config(
        page_title=str(config.app_title or "PubMed Literature Assistant"),
        page_icon="📚",
        layout="wide",
    )

    app_state.init_state(default_top_n=10)
    apply_app_styles()
    render_header(config)

    with st.expander("How to use", expanded=False):
        st.markdown(
            "- Ask a medical or health literature question.\n"
            "- Turn **Follow-up mode** on to rewrite short follow-ups with chat context.\n"
            "- Turn **Show papers** on if you want ranked source links in the UI.\n"
            f"- Adjust **Top-N papers** in the sidebar (current: **{app_state.get_top_n()}**).",
            unsafe_allow_html=False,
        )

    sidebar_action = render_sidebar(
        chats=app_state.get_recent_chats(limit=5),
        active_chat_id=app_state.get_active_chat_id(),
        top_n=app_state.get_top_n(),
    )
    app_state.set_top_n(sidebar_action["top_n"])

    if sidebar_action.get("switch_chat_id"):
        app_state.switch_chat(str(sidebar_action["switch_chat_id"]))
        _safe_rerun()
        return

    if sidebar_action.get("new_chat"):
        app_state.new_chat()
        _safe_rerun()
        return

    if sidebar_action.get("clear_chat"):
        active_chat_id = app_state.get_active_chat_id()
        app_state.clear_active_messages()
        try:
            clear_session_history(active_chat_id)
        except Exception:
            pass
        _safe_rerun()
        return

    if bool(getattr(config, "eval_mode", False)):
        chat_tab, eval_tab = st.tabs(["Chat", "Evaluation Dashboard"])
        with chat_tab:
            _render_chat_experience(config)
        with eval_tab:
            render_evaluation_dashboard(str(getattr(config, "eval_store_path")))
    else:
        _render_chat_experience(config)


if __name__ == "__main__":
    main()
