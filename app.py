from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from src import app_state
from eval.dashboard import render_evaluation_dashboard
from eval.evaluator import evaluate_turn, should_sample_query
from eval.store import EvalStore
from src.chat.router import invoke_chat_request, stream_chat_request
from src.core.config import ConfigValidationError, load_config
from src.history import clear_session_history
from src.logging_utils import log_event, setup_logging
from src.ui.metrics_dashboard import render_metrics_dashboard
from src.ui.formatters import beautify_text, strip_reframe_block
from src.ui.render import (
    auto_scroll,
    apply_app_styles,
    get_thinking_message,
    render_chat,
    render_header,
    render_message,
    render_sidebar,
)


def _safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - legacy streamlit
        st.experimental_rerun()


def _scroll_to_bottom(*, enabled: bool) -> None:
    auto_scroll(enabled=enabled)


def _consume_stream_response(stream, placeholder, *, auto_scroll_enabled: bool) -> tuple[str, dict]:
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
        if auto_scroll_enabled and chunk_count % 2 == 0:
            _scroll_to_bottom(enabled=True)
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


def _render_chat_experience(config) -> None:
    top_n = app_state.get_top_n()
    follow_up_mode = app_state.get_follow_up_mode()
    show_papers = app_state.get_show_papers()
    show_rewritten_query = app_state.get_show_rewritten_query()
    auto_scroll_enabled = app_state.get_auto_scroll()
    render_chat(
        app_state.get_active_messages(),
        top_n=top_n,
        show_papers=show_papers,
        show_rewritten_query=show_rewritten_query,
    )
    st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)

    user_input = st.chat_input(
        "Ask a medical or health question (e.g., oncology, gut health, neurology)"
    )
    if not user_input:
        return

    app_state.append_active_message({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input, unsafe_allow_html=False)
    _scroll_to_bottom(enabled=auto_scroll_enabled)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        loading_message = get_thinking_message(user_input)
        placeholder.markdown(loading_message, unsafe_allow_html=False)

        answer_text = ""
        final_payload: dict = {}
        session_id = app_state.get_active_chat_id()
        chat_messages = app_state.get_active_messages()[:-1]
        conversation_summary = app_state.get_conversation_summary()

        try:
            stream = stream_chat_request(
                query=user_input,
                session_id=session_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                chat_messages=chat_messages,
                show_papers=show_papers,
                conversation_summary=conversation_summary,
            )
            answer_text, final_payload = _consume_stream_response(
                stream,
                placeholder,
                auto_scroll_enabled=auto_scroll_enabled,
            )
        except Exception:
            final_payload = invoke_chat_request(
                query=user_input,
                session_id=session_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                chat_messages=chat_messages,
                show_papers=show_papers,
                conversation_summary=conversation_summary,
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
        placeholder.empty()

        rewritten_query = str(final_payload.get("rewritten_query", "") or "").strip()
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

        render_message(
            assistant_message,
            top_n=top_n,
            show_papers=show_papers,
            show_rewritten_query=show_rewritten_query,
            message_key=str(final_payload.get("request_id") or session_id),
        )

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
        app_state.update_conversation_summary(app_state.get_active_messages())
        _maybe_run_online_evaluation(config=config, query=user_input, payload=final_payload)
        _scroll_to_bottom(enabled=auto_scroll_enabled)


def main() -> None:
    load_dotenv(override=False)
    config = load_config()
    setup_logging(config.log_level)
    log_event("config.loaded", config=config.masked_summary())
    st.set_page_config(
        page_title=str(config.app_title or "PubMed Literature Assistant"),
        page_icon="📚",
        layout="wide",
    )

    try:
        config.require_valid()
    except ConfigValidationError as exc:
        apply_app_styles()
        render_header(config)
        st.error(str(exc))
        with st.expander("Resolved Config", expanded=False):
            st.json(config.masked_summary())
        return

    app_state.init_state(
        default_top_n=10,
        default_show_papers=False,
        default_show_rewritten_query=bool(config.show_rewritten_query),
        default_auto_scroll=bool(config.auto_scroll),
        default_follow_up_mode=True,
    )
    apply_app_styles()
    render_header(config)

    with st.expander("How to use", expanded=False):
        st.markdown(
            "- Ask a medical or health literature question.\n"
            "- Turn **Follow-up mode** on to rewrite short follow-ups with chat context.\n"
            "- Use the sidebar toggles for paper links, rewritten query display, and auto-scroll.\n"
            f"- Adjust **Top-N papers** in the sidebar (current: **{app_state.get_top_n()}**).",
            unsafe_allow_html=False,
        )
    with st.expander("Runtime Config", expanded=False):
        st.json(config.masked_summary())

    sidebar_action = render_sidebar(
        chats=app_state.get_recent_chats(limit=5),
        active_chat_id=app_state.get_active_chat_id(),
        top_n=app_state.get_top_n(),
        follow_up_mode=app_state.get_follow_up_mode(),
        show_papers=app_state.get_show_papers(),
        show_rewritten_query=app_state.get_show_rewritten_query(),
        auto_scroll_enabled=app_state.get_auto_scroll(),
    )
    app_state.set_top_n(sidebar_action["top_n"])
    app_state.set_follow_up_mode(bool(sidebar_action.get("follow_up_mode", True)))
    app_state.set_show_papers(bool(sidebar_action.get("show_papers", False)))
    app_state.set_show_rewritten_query(
        bool(sidebar_action.get("show_rewritten_query", config.show_rewritten_query))
    )
    app_state.set_auto_scroll(bool(sidebar_action.get("auto_scroll", config.auto_scroll)))

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

    if bool(getattr(config, "eval_mode", False)) and bool(getattr(config, "metrics_mode", False)):
        chat_tab, eval_tab, metrics_tab = st.tabs(["Chat", "Evaluation Dashboard", "Metrics"])
        with chat_tab:
            _render_chat_experience(config)
        with eval_tab:
            render_evaluation_dashboard(str(getattr(config, "eval_store_path")))
        with metrics_tab:
            render_metrics_dashboard(str(getattr(config, "metrics_store_path")))
    elif bool(getattr(config, "eval_mode", False)):
        chat_tab, eval_tab = st.tabs(["Chat", "Evaluation Dashboard"])
        with chat_tab:
            _render_chat_experience(config)
        with eval_tab:
            render_evaluation_dashboard(str(getattr(config, "eval_store_path")))
    elif bool(getattr(config, "metrics_mode", False)):
        chat_tab, metrics_tab = st.tabs(["Chat", "Metrics"])
        with chat_tab:
            _render_chat_experience(config)
        with metrics_tab:
            render_metrics_dashboard(str(getattr(config, "metrics_store_path")))
    else:
        _render_chat_experience(config)


if __name__ == "__main__":
    main()
