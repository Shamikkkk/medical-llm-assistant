from __future__ import annotations

from pathlib import Path
import shutil

import streamlit as st
from dotenv import load_dotenv

from src import app_state
from eval.dashboard import render_evaluation_dashboard
from eval.evaluator import evaluate_turn, should_sample_query
from eval.store import EvalStore
from src.chat.router import invoke_chat_request, stream_chat_request
from src.core.config import ConfigValidationError, load_config
from src.history import clear_session_history, replace_session_history
from src.integrations.storage import (
    clear_answer_cache,
    clear_query_result_caches,
    resolve_compute_device,
)
from src.logging_utils import log_event, setup_logging
from src.ui.metrics_dashboard import render_metrics_dashboard
from src.ui.formatters import (
    beautify_text,
    export_branch_json,
    export_branch_markdown,
    strip_reframe_block,
)
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
    if bool(payload.get("answer_cache_hit", False)):
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


def _sync_active_history() -> None:
    replace_session_history(
        app_state.get_active_chat_id(),
        app_state.get_active_branch_id(),
        app_state.get_active_history_messages(),
    )


def _apply_compute_device_selection() -> None:
    preference = app_state.get_compute_device_preference()
    effective_device, warning = resolve_compute_device(preference)
    app_state.set_compute_device(preference, effective_device, warning)


def _build_export_payloads() -> tuple[str, str]:
    branch = app_state.get_active_branch_record()
    messages = app_state.get_active_messages()
    return (
        export_branch_markdown(
            chat_title=app_state.get_active_chat_title(),
            branch_title=str(branch.get("title", "") or "Conversation branch"),
            branch_id=str(branch.get("branch_id", "") or "main"),
            parent_branch_id=str(branch.get("parent_branch_id", "") or ""),
            messages=messages,
        ),
        export_branch_json(
            chat_id=app_state.get_active_chat_id(),
            chat_title=app_state.get_active_chat_title(),
            branch=branch,
            messages=messages,
        ),
    )


def _clear_paper_cache(data_dir: Path) -> int:
    papers_dir = data_dir / "papers"
    if not papers_dir.exists():
        return 0
    removed = sum(1 for _ in papers_dir.rglob("*"))
    shutil.rmtree(papers_dir, ignore_errors=True)
    return removed


def _build_assistant_message(final_payload: dict, answer_text: str) -> dict:
    rewritten_query = str(final_payload.get("rewritten_query", "") or "").strip()
    status = str(final_payload.get("status", "answered"))
    message = {
        "role": "assistant",
        "content": answer_text,
        "sources": final_payload.get("sources", []) or [],
        "retrieved_contexts": final_payload.get("retrieved_contexts", []) or [],
        "pubmed_query": final_payload.get("pubmed_query", ""),
        "reranker_active": final_payload.get("reranker_active", False),
        "status": status,
        "validation_warning": final_payload.get("validation_warning", ""),
        "validation_issues": final_payload.get("validation_issues", []) or [],
        "source_count_note": final_payload.get("source_count_note", ""),
        "rewritten_query": rewritten_query,
        "answer_cache_hit": bool(final_payload.get("answer_cache_hit", False)),
        "answer_cache_created_at": final_payload.get("answer_cache_created_at", ""),
        "answer_cache_similarity": final_payload.get("answer_cache_similarity", 0.0),
        "answer_cache_note": final_payload.get("answer_cache_note", ""),
        "timings": final_payload.get("timings", {}) or {},
    }
    if status in {"out_of_scope", "smalltalk"}:
        return {
            "role": "assistant",
            "content": answer_text,
            "status": status,
            "timings": final_payload.get("timings", {}) or {},
        }
    return message


def _execute_assistant_turn(
    *,
    config,
    user_input: str,
    render_user_bubble: bool,
) -> None:
    top_n = app_state.get_top_n()
    follow_up_mode = app_state.get_follow_up_mode()
    show_papers = app_state.get_show_papers()
    show_rewritten_query = app_state.get_show_rewritten_query()
    auto_scroll_enabled = app_state.get_auto_scroll()
    session_id = app_state.get_active_chat_id()
    branch_id = app_state.get_active_branch_id()
    compute_device = app_state.get_effective_compute_device()

    if render_user_bubble:
        with st.chat_message("user"):
            st.markdown(user_input, unsafe_allow_html=False)
    _scroll_to_bottom(enabled=auto_scroll_enabled)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        loading_message = get_thinking_message(user_input)
        placeholder.markdown(loading_message, unsafe_allow_html=False)

        answer_text = ""
        final_payload: dict = {}
        chat_messages = app_state.get_active_messages()[:-1]
        conversation_summary = app_state.get_conversation_summary()

        try:
            stream = stream_chat_request(
                query=user_input,
                session_id=session_id,
                branch_id=branch_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                chat_messages=chat_messages,
                show_papers=show_papers,
                conversation_summary=conversation_summary,
                compute_device=compute_device,
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
                branch_id=branch_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                chat_messages=chat_messages,
                show_papers=show_papers,
                conversation_summary=conversation_summary,
                compute_device=compute_device,
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

        assistant_message = _build_assistant_message(final_payload, answer_text)
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
        answer_cache_meta = {}
        if bool(final_payload.get("answer_cache_hit", False)):
            answer_cache_meta = {
                "created_at": str(final_payload.get("answer_cache_created_at", "") or ""),
                "similarity": float(final_payload.get("answer_cache_similarity", 0.0) or 0.0),
                "query": str(final_payload.get("answer_cache_query", "") or ""),
            }
        app_state.update_active_context_state(
            last_topic_summary=str(final_payload.get("last_topic_summary", "") or ""),
            last_retrieved_sources=new_sources,
            last_response_metrics=final_payload.get("timings", {}) or {},
            last_answer_cache=answer_cache_meta,
        )
        app_state.update_conversation_summary(app_state.get_active_messages())
        _maybe_run_online_evaluation(config=config, query=user_input, payload=final_payload)
        _scroll_to_bottom(enabled=auto_scroll_enabled)


def _render_edit_branch_composer() -> None:
    edit_target = app_state.get_edit_target()
    if not edit_target:
        return
    current_content = str(edit_target.get("content", "") or "")
    st.markdown("### Edit Previous Prompt", unsafe_allow_html=False)
    with st.form("branch_edit_form", clear_on_submit=False):
        edited_text = st.text_area(
            "Edited user prompt",
            value=current_content,
            height=120,
        )
        create_branch = st.form_submit_button("Create branch from edit")
        cancel = st.form_submit_button("Cancel")
    if cancel:
        app_state.clear_edit_target()
        _safe_rerun()
        return
    if create_branch:
        branch_id = app_state.create_branch_from_edit(
            int(edit_target.get("message_index", 0)),
            edited_text,
        )
        app_state.set_pending_branch_submission(branch_id=branch_id, query=edited_text)
        _sync_active_history()
        _safe_rerun()


def _render_chat_experience(config) -> None:
    top_n = app_state.get_top_n()
    follow_up_mode = app_state.get_follow_up_mode()
    show_papers = app_state.get_show_papers()
    show_rewritten_query = app_state.get_show_rewritten_query()
    auto_scroll_enabled = app_state.get_auto_scroll()

    chat_action = render_chat(
        app_state.get_active_messages(),
        top_n=top_n,
        show_papers=show_papers,
        show_rewritten_query=show_rewritten_query,
    )
    if chat_action.get("edit_message_index") is not None:
        target_index = int(chat_action["edit_message_index"])
        messages = app_state.get_active_messages()
        app_state.set_edit_target(target_index, str(messages[target_index].get("content", "") or ""))
        _safe_rerun()
        return

    st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)
    _render_edit_branch_composer()

    pending_branch_submission = app_state.get_pending_branch_submission()
    if pending_branch_submission and str(pending_branch_submission.get("branch_id", "")) == app_state.get_active_branch_id():
        _execute_assistant_turn(
            config=config,
            user_input=str(pending_branch_submission.get("query", "") or ""),
            render_user_bubble=False,
        )
        app_state.clear_pending_branch_submission()
        return

    if app_state.get_edit_target():
        return

    user_input = st.chat_input(
        "Ask a medical or health question (e.g., oncology, gut health, neurology)"
    )
    if not user_input:
        return

    app_state.append_active_message({"role": "user", "content": user_input})
    _execute_assistant_turn(
        config=config,
        user_input=user_input,
        render_user_bubble=True,
    )


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
        default_compute_device_preference="auto",
    )
    _apply_compute_device_selection()
    _sync_active_history()

    apply_app_styles()
    render_header(config)

    with st.expander("How to use", expanded=False):
        st.markdown(
            "- Ask a medical or health literature question.\n"
            "- Turn **Follow-up mode** on to rewrite short follow-ups with chat context.\n"
            "- Use **Compute device** to prefer CPU or GPU for local embeddings and validation.\n"
            "- Edit any previous user prompt to create a new branch from that point.\n"
            "- Similar-query answers can be reused from cache when the runtime fingerprint still matches.\n"
            f"- Adjust **Top-N papers** in the sidebar (current: **{app_state.get_top_n()}**).",
            unsafe_allow_html=False,
        )
    with st.expander("Runtime Config", expanded=False):
        st.json(config.masked_summary())

    export_markdown, export_json = _build_export_payloads()
    sidebar_action = render_sidebar(
        chats=app_state.get_recent_chats(limit=5),
        active_chat_id=app_state.get_active_chat_id(),
        branches=app_state.get_branches_for_active_chat(),
        active_branch_id=app_state.get_active_branch_id(),
        top_n=app_state.get_top_n(),
        follow_up_mode=app_state.get_follow_up_mode(),
        show_papers=app_state.get_show_papers(),
        show_rewritten_query=app_state.get_show_rewritten_query(),
        auto_scroll_enabled=app_state.get_auto_scroll(),
        compute_device_preference=app_state.get_compute_device_preference(),
        effective_compute_device=app_state.get_effective_compute_device(),
        compute_device_warning=app_state.get_compute_device_warning(),
        export_markdown=export_markdown,
        export_json=export_json,
        last_response_metrics=app_state.get_active_context_state().get("last_response_metrics", {}) or {},
    )
    app_state.set_top_n(sidebar_action["top_n"])
    app_state.set_follow_up_mode(bool(sidebar_action.get("follow_up_mode", True)))
    app_state.set_show_papers(bool(sidebar_action.get("show_papers", False)))
    app_state.set_show_rewritten_query(
        bool(sidebar_action.get("show_rewritten_query", config.show_rewritten_query))
    )
    app_state.set_auto_scroll(bool(sidebar_action.get("auto_scroll", config.auto_scroll)))
    if sidebar_action.get("compute_device_preference") != app_state.get_compute_device_preference():
        app_state.set_compute_device(
            str(sidebar_action.get("compute_device_preference", "auto") or "auto"),
            app_state.get_effective_compute_device(),
            app_state.get_compute_device_warning(),
        )
    _apply_compute_device_selection()

    if sidebar_action.get("switch_chat_id"):
        app_state.switch_chat(str(sidebar_action["switch_chat_id"]))
        _sync_active_history()
        _safe_rerun()
        return

    if sidebar_action.get("switch_branch_id"):
        app_state.switch_branch(str(sidebar_action["switch_branch_id"]))
        _sync_active_history()
        _safe_rerun()
        return

    if sidebar_action.get("new_chat"):
        app_state.new_chat()
        _sync_active_history()
        _safe_rerun()
        return

    if sidebar_action.get("clear_chat"):
        active_chat_id = app_state.get_active_chat_id()
        active_branch_id = app_state.get_active_branch_id()
        app_state.clear_active_messages()
        try:
            clear_session_history(active_chat_id, active_branch_id)
        except Exception:
            pass
        _safe_rerun()
        return

    if sidebar_action.get("clear_query_cache"):
        removed = clear_query_result_caches(
            str(config.data_dir / "chroma"),
            embeddings_device=app_state.get_effective_compute_device(),
        )
        with st.sidebar:
            st.success(f"Cleared {removed} query cache entries.")

    if sidebar_action.get("clear_answer_cache"):
        removed = clear_answer_cache(
            str(config.data_dir / "chroma"),
            embeddings_device=app_state.get_effective_compute_device(),
        )
        with st.sidebar:
            st.success(f"Cleared {removed} answer cache entries.")

    if sidebar_action.get("clear_paper_cache"):
        removed = _clear_paper_cache(Path(config.data_dir))
        with st.sidebar:
            st.success(f"Cleared {removed} paper cache artifacts.")

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
