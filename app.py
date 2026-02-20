from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src import app_state
from eval.dashboard import render_evaluation_dashboard
from eval.evaluator import evaluate_turn, should_sample_query
from eval.store import EvalStore
from src.chat.router import (
    ingest_link_for_selected_paper,
    ingest_uploaded_pdf_for_selected_paper,
    invoke_chat_request,
    stream_chat_request,
)
from src.core.config import load_config
from src.history import clear_session_history
from src.logging_utils import setup_logging
from src.ui.formatters import beautify_text, pubmed_url, strip_reframe_block
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


def _button_stretch(label: str, *, key: str | None = None) -> bool:
    try:
        return st.button(label, key=key, width="stretch")
    except TypeError:
        return st.button(label, key=key, use_container_width=True)


def _scroll_to_bottom() -> None:
    components.html(
        """
        <script>
        const root = window.parent.document;
        const target = root.getElementById("chat-bottom");
        if (target) {
            target.scrollIntoView({behavior: "smooth", block: "end"});
        }
        </script>
        """,
        height=0,
    )


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
    mode = "online_paper_focus" if bool(payload.get("paper_focus_mode")) else "online"
    record = evaluate_turn(
        query=query,
        answer=answer,
        contexts=contexts,
        sources=sources,
        mode=mode,
    )
    record["paper_focus_mode"] = bool(payload.get("paper_focus_mode", False))
    record["selected_pmid"] = str(payload.get("paper_focus_pmid", "") or "")
    EvalStore(getattr(config, "eval_store_path")).append(record)


def _render_followup_controls(config) -> None:
    context_state = app_state.get_active_context_state()
    follow_up_mode = bool(context_state.get("follow_up_mode", False))
    selected_paper = app_state.get_selected_paper()
    left, right = st.columns([1.4, 1])
    with left:
        next_mode = st.toggle("Follow-up mode", value=follow_up_mode)
        if next_mode != follow_up_mode:
            app_state.set_follow_up_mode(next_mode)
            follow_up_mode = next_mode
    with right:
        if selected_paper:
            pmid = str(selected_paper.get("pmid", "") or "")
            title = str(selected_paper.get("title", "") or "")
            st.caption(f"Paper Focus: {pmid} {title[:72]}")
            if _button_stretch(
                "Clear paper focus", key=f"clear_focus_{app_state.get_active_chat_id()}"
            ):
                app_state.set_selected_paper(None)
                _safe_rerun()

    if selected_paper:
        default_link = str(selected_paper.get("fulltext_url", "") or "").strip()
        if default_link:
            st.caption(f"Detected full-text link: {default_link}")
        custom_link = st.text_input(
            "Read from link (OA/public URL)",
            value=default_link,
            key=f"paper_link_{app_state.get_active_chat_id()}_{selected_paper.get('pmid', '')}",
            placeholder="https://...",
        )
        if _button_stretch(
            "Fetch & Ingest from link",
            key=f"paper_link_ingest_{app_state.get_active_chat_id()}_{selected_paper.get('pmid', '')}",
        ):
            result = ingest_link_for_selected_paper(
                selected_paper=selected_paper,
                link_url=custom_link,
                data_dir=config.data_dir,
            )
            if result.get("ok"):
                st.success(str(result.get("message", "Link content ingested.")))
                paper = result.get("paper", {}) or {}
                if isinstance(paper, dict):
                    app_state.set_selected_paper(
                        {
                            "pmid": paper.get("pmid", selected_paper.get("pmid", "")),
                            "title": paper.get("title", selected_paper.get("title", "")),
                            "journal": paper.get("journal", selected_paper.get("journal", "")),
                            "year": paper.get("year", selected_paper.get("year", "")),
                            "doi": paper.get("doi", selected_paper.get("doi", "")),
                            "pmcid": paper.get("pmcid", selected_paper.get("pmcid", "")),
                            "fulltext_url": paper.get(
                                "fulltext_url", selected_paper.get("fulltext_url", "")
                            ),
                        }
                    )
                _safe_rerun()
            else:
                st.warning(str(result.get("message", "Could not ingest from the provided link.")))

        uploaded_pdf = st.file_uploader(
            "Upload PDF for focused paper (optional)",
            type=["pdf"],
            key=f"paper_pdf_{app_state.get_active_chat_id()}_{selected_paper.get('pmid', '')}",
        )
        if uploaded_pdf is not None:
            result = ingest_uploaded_pdf_for_selected_paper(
                selected_paper=selected_paper,
                uploaded_bytes=uploaded_pdf.getvalue(),
                file_name=uploaded_pdf.name,
                data_dir=config.data_dir,
            )
            if result.get("ok"):
                st.success(str(result.get("message", "PDF ingested.")))
                paper = result.get("paper", {}) or {}
                if isinstance(paper, dict):
                    app_state.set_selected_paper(
                        {
                            "pmid": paper.get("pmid", selected_paper.get("pmid", "")),
                            "title": paper.get("title", selected_paper.get("title", "")),
                            "journal": paper.get("journal", selected_paper.get("journal", "")),
                            "year": paper.get("year", selected_paper.get("year", "")),
                            "doi": paper.get("doi", selected_paper.get("doi", "")),
                            "pmcid": paper.get("pmcid", selected_paper.get("pmcid", "")),
                            "fulltext_url": paper.get(
                                "fulltext_url", selected_paper.get("fulltext_url", "")
                            ),
                        }
                    )
                _safe_rerun()
            else:
                st.warning(str(result.get("message", "Could not ingest uploaded PDF.")))


def _render_suggested_papers_panel() -> None:
    context_state = app_state.get_active_context_state()
    sources = context_state.get("last_retrieved_sources", []) or []
    if not isinstance(sources, list) or not sources:
        return
    selected = app_state.get_selected_paper() or {}
    selected_pmid = str(selected.get("pmid", "") or "").strip()
    st.markdown("### Suggested Papers", unsafe_allow_html=False)
    st.caption("Select one paper to keep follow-up answers focused on that paper.")
    active_chat_id = app_state.get_active_chat_id()
    for index, source in enumerate(sources, start=1):
        pmid = str(source.get("pmid", "") or "").strip()
        title = str(source.get("title", "") or "Untitled").strip()
        journal = str(source.get("journal", "") or "").strip()
        year = str(source.get("year", "") or "").strip()
        doi = str(source.get("doi", "") or "").strip()
        fulltext_url = str(source.get("fulltext_url", "") or "").strip()
        is_selected = bool(pmid and pmid == selected_pmid)
        st.markdown(f"**{index}) {title}**", unsafe_allow_html=False)
        st.markdown(f"PMID: `{pmid or 'N/A'}`", unsafe_allow_html=False)
        if doi:
            st.markdown(f"DOI: `{doi}`", unsafe_allow_html=False)
        if journal or year:
            meta = f"{journal} ({year})" if journal and year else journal or year
            st.markdown(meta, unsafe_allow_html=False)
        if pmid:
            st.markdown(f"[PubMed]({pubmed_url(pmid)})", unsafe_allow_html=False)
        if fulltext_url:
            st.markdown(f"[Full Text Link]({fulltext_url})", unsafe_allow_html=False)
        if is_selected:
            st.caption("Selected for Paper Focus")
        toggle_label = "Clear focus for this paper" if is_selected else "Focus this paper"
        if _button_stretch(
            toggle_label,
            key=f"use_paper_{active_chat_id}_{pmid or index}",
        ):
            if is_selected:
                app_state.set_selected_paper(None)
            else:
                app_state.set_selected_paper(
                    {
                        "pmid": pmid,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                        "pmcid": str(source.get("pmcid", "") or "").strip(),
                        "fulltext_url": fulltext_url,
                    }
                )
            _safe_rerun()
        st.markdown("---", unsafe_allow_html=False)


def _render_chat_experience(config) -> None:
    top_n = app_state.get_top_n()
    _render_followup_controls(config)
    render_chat(app_state.get_active_messages(), top_n=top_n)
    _render_suggested_papers_panel()
    st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask a cardiovascular question")
    if not user_input:
        return

    app_state.append_active_message({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input, unsafe_allow_html=False)
    _scroll_to_bottom()

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Stay steady, breathe deep...", unsafe_allow_html=False)

        answer_text = ""
        final_payload: dict = {}
        session_id = app_state.get_active_chat_id()
        context_state = app_state.get_active_context_state()
        selected_paper = app_state.get_selected_paper()
        follow_up_mode = bool(context_state.get("follow_up_mode", False))
        chat_messages = app_state.get_active_messages()[:-1]

        try:
            stream = stream_chat_request(
                query=user_input,
                session_id=session_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                selected_paper=selected_paper,
                chat_messages=chat_messages,
                data_dir=config.data_dir,
            )
            answer_text, final_payload = _consume_stream_response(stream, placeholder)
        except Exception:
            final_payload = invoke_chat_request(
                query=user_input,
                session_id=session_id,
                top_n=top_n,
                agent_mode=bool(getattr(config, "agent_mode", False)),
                follow_up_mode=follow_up_mode,
                selected_paper=selected_paper,
                chat_messages=chat_messages,
                data_dir=config.data_dir,
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
        if final_payload.get("paper_focus_notice"):
            st.info(str(final_payload.get("paper_focus_notice")))
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
            "paper_focus_mode": bool(final_payload.get("paper_focus_mode", False)),
            "paper_focus_notice": str(final_payload.get("paper_focus_notice", "") or ""),
            "paper_focus_pmid": str(final_payload.get("paper_focus_pmid", "") or ""),
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

        if status == "answered":
            render_ranked_sources(assistant_message.get("sources", []) or [], top_n=top_n)
        _scroll_to_bottom()


def main() -> None:
    load_dotenv(override=False)
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
