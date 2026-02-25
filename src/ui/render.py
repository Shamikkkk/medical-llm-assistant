from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping

import streamlit as st
import streamlit.components.v1 as components

from src.core.config import AppConfig
from src.types import SourceItem
from src.ui.formatters import beautify_text, doi_url, pubmed_url, strip_reframe_block


def auto_scroll() -> None:
    """Scroll the parent Streamlit page toward the latest chat content."""
    components.html(
        """
        <script>
        const root = window.parent.document;
        const target = root.getElementById("chat-bottom");
        const doScroll = () => {
          if (target) {
            target.scrollIntoView({ behavior: "smooth", block: "end" });
          } else {
            window.parent.scrollTo({
              top: root.body ? root.body.scrollHeight : 0,
              behavior: "smooth",
            });
          }
        };
        requestAnimationFrame(doScroll);
        </script>
        """,
        height=0,
    )


def apply_app_styles() -> None:
    st.markdown(
        """
<style>
    .main .block-container {
        max-width: 1040px;
        padding-top: 1.2rem;
        padding-bottom: 8rem;
    }
    .cardio-header-row {
        margin-bottom: 0.35rem;
    }
    .cardio-title-wrap {
        text-align: center;
        margin-bottom: 0.35rem;
    }
    .cardio-title-wrap h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .cardio-title-wrap p {
        margin: 0.35rem 0 0.4rem 0;
        font-size: 1rem;
        opacity: 0.85;
    }
    .cardio-divider {
        border: 0;
        border-top: 1px solid rgba(127, 127, 127, 0.3);
        margin: 0.45rem auto 1rem auto;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.45rem 0.65rem 0.5rem 0.65rem;
        border: 1px solid rgba(127, 127, 127, 0.28);
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    .theme-toggle-wrap {
        display: flex;
        justify-content: flex-end;
        align-items: flex-start;
        min-height: 3rem;
        margin-top: 0.15rem;
    }
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 100;
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
        backdrop-filter: blur(3px);
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header(config: AppConfig) -> None:
    title = str(config.app_title or "PubMed Literature Assistant")
    subtitle = str(
        config.app_description or "Medical literature assistant grounded in PubMed abstracts"
    )
    _, middle_col, right_col = st.columns([1, 3, 1])
    with middle_col:
        st.markdown(
            "\n".join(
                [
                    '<div class="cardio-title-wrap">',
                    f"<h1>{title}</h1>",
                    f"<p>{subtitle}</p>",
                    "</div>",
                ]
            ),
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown('<div class="theme-toggle-wrap">', unsafe_allow_html=True)
        _render_theme_toggle()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="cardio-divider" />', unsafe_allow_html=True)


def render_sidebar(
    *,
    chats: list[dict],
    active_chat_id: str,
    top_n: int,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "top_n": top_n,
        "new_chat": False,
        "clear_chat": False,
        "switch_chat_id": None,
    }

    with st.sidebar:
        st.markdown("### Controls", unsafe_allow_html=False)
        action["top_n"] = st.slider(
            "Top-N papers",
            min_value=1,
            max_value=10,
            value=int(top_n),
            step=1,
        )

        if _button_stretch("New Chat"):
            action["new_chat"] = True
        if _button_stretch("Clear active chat"):
            action["clear_chat"] = True

        st.markdown("### Recent Chats", unsafe_allow_html=False)
        st.caption("Last 5 chats")
        recent = sorted(
            chats,
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )[:5]
        if not recent:
            st.caption("No conversations yet.")
            return action

        for chat in recent:
            chat_id = str(chat.get("chat_id", ""))
            title = str(chat.get("title", "") or "New conversation")
            created_at = _format_timestamp(str(chat.get("created_at", "")))
            prefix = "* " if chat_id == active_chat_id else ""
            label = f"{prefix}{title} ({created_at})" if created_at else f"{prefix}{title}"
            if _button_stretch(label, key=f"chat_switch_{chat_id}"):
                action["switch_chat_id"] = chat_id

    return action


def render_chat(messages: list[dict], *, top_n: int, show_papers: bool) -> None:
    for idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            render_message(
                message,
                top_n=top_n,
                show_papers=show_papers,
                message_key=f"assistant_{idx}",
            )


def render_message(
    message: Mapping[str, Any],
    *,
    top_n: int,
    show_papers: bool = True,
    message_key: str = "",
) -> None:
    role = message.get("role")
    content = strip_reframe_block(str(message.get("content", "") or ""))
    if role == "assistant":
        st.markdown(beautify_text(content), unsafe_allow_html=False)
        rewritten_query = str(message.get("rewritten_query", "") or "").strip()
        if rewritten_query:
            st.caption(f"Follow-up rewrite: {rewritten_query}")
        warning = str(message.get("validation_warning", "") or "").strip()
        issues = message.get("validation_issues", []) or []
        if warning:
            st.warning(warning)
        if issues:
            with st.expander("Validation details", expanded=False):
                for issue in issues:
                    st.markdown(f"- {issue}", unsafe_allow_html=False)

        status = str(message.get("status", "answered"))
        if should_render_sources(status=status, show_papers=show_papers):
            render_ranked_sources(message.get("sources", []) or [], top_n=top_n)
        _render_copy_button(content, key=f"copy_{message_key}")
        return
    st.markdown(content, unsafe_allow_html=False)


def should_render_sources(*, status: str, show_papers: bool) -> bool:
    return bool(show_papers and str(status) == "answered")


def render_ranked_sources(sources: list[dict], top_n: int) -> None:
    ranked_sources = _rank_sources(sources, limit=top_n)
    if not ranked_sources:
        return

    st.markdown(f"### Top {top_n} ranked sources", unsafe_allow_html=False)
    for source in ranked_sources:
        render_source_item(source)
        st.markdown("", unsafe_allow_html=False)


def render_sources(sources: list[dict], *, top_n: int) -> None:
    """Backward-compatible wrapper."""
    render_ranked_sources(sources, top_n=top_n)


def render_source_item(source: SourceItem) -> None:
    rank = int(source.get("rank", 0))
    title = str(source.get("title", "") or "Untitled").strip()
    pmid = str(source.get("pmid", "") or "").strip()
    pmid_display = pmid if pmid else "N/A"
    journal = str(source.get("journal", "") or "").strip()
    year = str(source.get("year", "") or "").strip()
    doi = str(source.get("doi", "") or "").strip()

    meta = ""
    if journal and year:
        meta = f"Journal: {journal} ({year})"
    elif journal:
        meta = f"Journal: {journal}"
    elif year:
        meta = f"Year: {year}"

    markdown_lines = [f"**{rank}) {title}**", f"PMID: `{pmid_display}`"]
    if meta:
        markdown_lines.append(meta)
    if doi:
        markdown_lines.append(f"[DOI]({doi_url(doi)})")
    if pmid:
        markdown_lines.append(f"[PubMed]({pubmed_url(pmid)})")
    st.markdown("\n\n".join(markdown_lines), unsafe_allow_html=False)


def _rank_sources(items: list[dict], limit: int) -> list[SourceItem]:
    ranked: list[SourceItem] = []
    seen_pmids: set[str] = set()
    ordered_items = _sort_sources(items)

    for item in ordered_items:
        pmid = str(item.get("pmid", "") or "").strip()
        if pmid and pmid in seen_pmids:
            continue
        if pmid:
            seen_pmids.add(pmid)
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "pmid": pmid,
                "title": str(item.get("title", "") or ""),
                "journal": str(item.get("journal", "") or ""),
                "year": str(item.get("year", "") or ""),
                "doi": str(item.get("doi", "") or ""),
                "pmcid": str(item.get("pmcid", "") or ""),
            }
        )
        if len(ranked) >= limit:
            break
    return ranked


def _sort_sources(items: list[dict]) -> list[dict]:
    has_rank = any(item.get("rank") not in (None, "") for item in items)
    if not has_rank:
        return list(items)

    def _rank_key(item: dict) -> tuple[int, int]:
        value = item.get("rank")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 10**6
        return (parsed, 0)

    return sorted(items, key=_rank_key)


def _format_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M")


def _render_theme_toggle() -> None:
    try:
        from streamlit_plugins.components.theme_changer import st_theme_changer

        try:
            st_theme_changer(render_mode="pills")
        except TypeError:
            st_theme_changer()
    except Exception:
        st.caption("Theme: use menu -> Settings -> Theme")


def _button_stretch(label: str, key: str | None = None) -> bool:
    try:
        return st.button(label, key=key, width="stretch")
    except TypeError:
        return st.button(label, key=key, use_container_width=True)


def _render_copy_button(text: str, *, key: str) -> None:
    if not key:
        key = f"copy_{abs(hash(text))}"
    payload = json.dumps(str(text or ""))
    components.html(
        f"""
        <div style="display:flex;justify-content:flex-end;">
            <button id="{key}" style="font-size:0.8rem;padding:0.25rem 0.5rem;border-radius:6px;border:1px solid #aaa;cursor:pointer;"
                onclick='(async () => {{
                    const value = {payload};
                    try {{
                        await navigator.clipboard.writeText(value);
                        this.innerText = "Copied";
                    }} catch (e) {{
                        this.innerText = "Copy failed";
                    }}
                }})()'
            >Copy response</button>
        </div>
        """,
        height=40,
    )
