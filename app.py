from __future__ import annotations

import streamlit as st

from src import app_state
from src.core.config import load_config
from src.core.pipeline import invoke_chat, stream_chat
from src.history import clear_session_history
from src.logging_utils import setup_logging
from src.ui.formatters import beautify_text, strip_reframe_block
from src.ui.render import (
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


def _consume_stream_response(stream, placeholder) -> tuple[str, dict]:
    answer_text = ""
    final_payload: dict = {}
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
    return answer_text, final_payload


def main() -> None:
    config = load_config()
    setup_logging(config.log_level)
    st.set_page_config(
        page_title="Cardio PubMed Assistant",
        page_icon=":anatomical_heart:",
        layout="wide",
    )

    app_state.init_state(default_top_n=10)
    apply_app_styles()
    render_header(config)

    with st.expander("How to use", expanded=False):
        st.markdown(
            "- Ask a cardiovascular or cardiopulmonary overlap question.\n"
            f"- Adjust **Top-N papers** in the sidebar (current: **{app_state.get_top_n()}**).\n"
            "- Review cited PMIDs and open PubMed links for evidence details.",
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

    top_n = app_state.get_top_n()
    render_chat(app_state.get_active_messages(), top_n=top_n)

    user_input = st.chat_input("Ask a cardiovascular question")
    if not user_input:
        return

    app_state.append_active_message({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input, unsafe_allow_html=False)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Stay steady, breathe deep...", unsafe_allow_html=False)

        answer_text = ""
        final_payload: dict = {}
        session_id = app_state.get_active_chat_id()

        try:
            stream = stream_chat(user_input, session_id=session_id, top_n=top_n)
            answer_text, final_payload = _consume_stream_response(stream, placeholder)
        except Exception:
            final_payload = invoke_chat(user_input, session_id=session_id, top_n=top_n)
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
        placeholder.markdown(beautify_text(answer_text), unsafe_allow_html=False)

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

        if status == "answered":
            render_ranked_sources(assistant_message.get("sources", []) or [], top_n=top_n)


if __name__ == "__main__":
    main()
