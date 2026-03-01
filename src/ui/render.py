from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping

import streamlit as st
import streamlit.components.v1 as components

from src.core.config import AppConfig
from src.types import SourceItem
from src.ui.formatters import beautify_text, doi_url, pubmed_url, strip_reframe_block

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gi": ("gi", "gastro", "hepatic", "liver", "bowel", "colitis", "pancrea"),
    "neuro": ("neuro", "brain", "stroke", "seizure", "parkinson", "alzheim", "cognitive"),
    "cardio": ("cardio", "heart", "atrial", "ventric", "coronary", "hypertension", "hf"),
    "oncology": (
        "oncology",
        "cancer",
        "tumor",
        "tumour",
        "chemo",
        "metast",
        "temozolomide",
        "glioblastoma",
    ),
    "pulmonary": ("pulmonary", "copd", "asthma", "lung", "respirat", "pneumonia"),
}

THINKING_MESSAGES: dict[str, tuple[str, ...]] = {
    "gi": ("Digesting the literature...", "Checking the gut-level evidence..."),
    "neuro": ("Firing up neurons...", "Tracing the evidence pathways..."),
    "cardio": ("Following the heartbeat of the evidence...", "Cross-checking the cardiac literature..."),
    "oncology": ("Scanning the evidence landscape...", "Reviewing the treatment horizon..."),
    "pulmonary": ("Taking a deep breath through the abstracts...", "Surveying the pulmonary evidence..."),
    "general": ("Thinking...", "Reviewing the evidence...", "Pulling the relevant abstracts..."),
}


def auto_scroll(*, enabled: bool = True) -> None:
    """Scroll the parent Streamlit page toward the latest chat content."""
    components.html(build_auto_scroll_html(enabled=enabled), height=0)


def build_auto_scroll_html(*, enabled: bool) -> str:
    if not enabled:
        return "<div id='chat-autoscroll-disabled'></div>"
    return """
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
    """


def apply_app_styles() -> None:
    st.markdown(
        """
<style>
    #MainMenu {
        visibility: hidden;
    }
    button[title="Main menu"] {
        display: none !important;
    }
    .main .block-container {
        max-width: 1040px;
        padding-top: 1.2rem;
        padding-bottom: 8rem;
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
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        z-index: 100;
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
        backdrop-filter: blur(3px);
    }
    .cache-pill {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        border-radius: 999px;
        border: 1px solid rgba(127, 127, 127, 0.35);
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
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
    st.markdown('<hr class="cardio-divider" />', unsafe_allow_html=True)


def render_sidebar(
    *,
    chats: list[dict],
    active_chat_id: str,
    branches: list[dict] | None = None,
    active_branch_id: str = "main",
    top_n: int,
    follow_up_mode: bool,
    show_papers: bool,
    show_rewritten_query: bool,
    auto_scroll_enabled: bool,
    compute_device_preference: str = "auto",
    effective_compute_device: str = "cpu",
    compute_device_warning: str = "",
    export_markdown: str = "",
    export_json: str = "",
    last_response_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "top_n": top_n,
        "new_chat": False,
        "clear_chat": False,
        "switch_chat_id": None,
        "follow_up_mode": follow_up_mode,
        "show_papers": show_papers,
        "show_rewritten_query": show_rewritten_query,
        "auto_scroll": auto_scroll_enabled,
        "compute_device_preference": compute_device_preference,
        "switch_branch_id": None,
        "clear_query_cache": False,
        "clear_answer_cache": False,
        "clear_paper_cache": False,
    }

    branch_rows = list(branches or [])
    branch_index = max(
        0,
        next(
            (idx for idx, branch in enumerate(branch_rows) if str(branch.get("branch_id", "")) == active_branch_id),
            0,
        ),
    )

    with st.sidebar:
        st.markdown("### Controls", unsafe_allow_html=False)
        action["top_n"] = st.slider(
            "Top-N papers",
            min_value=1,
            max_value=10,
            value=int(top_n),
            step=1,
        )
        action["follow_up_mode"] = st.toggle(
            "Follow-up mode",
            value=bool(follow_up_mode),
            help="Use chat context to rewrite follow-up questions before retrieval.",
        )
        action["show_papers"] = st.toggle(
            "Show papers",
            value=bool(show_papers),
            help="Show ranked paper titles and links in the chat response.",
        )
        action["show_rewritten_query"] = st.toggle(
            "Show rewritten query",
            value=bool(show_rewritten_query),
            help="Display the interpreted standalone query for follow-up questions.",
        )
        action["auto_scroll"] = st.toggle(
            "Auto-scroll",
            value=bool(auto_scroll_enabled),
            help="Keep the newest streamed content in view.",
        )

        device_options = {"Auto": "auto", "CPU": "cpu", "GPU": "gpu"}
        reverse_device_options = {value: label for label, value in device_options.items()}
        selected_device = st.selectbox(
            "Compute device",
            options=list(device_options.keys()),
            index=list(device_options.keys()).index(
                reverse_device_options.get(str(compute_device_preference or "auto"), "Auto")
            ),
            help="Used for local embeddings and the optional validator. Remote NVIDIA generation is unchanged.",
        )
        action["compute_device_preference"] = device_options[selected_device]
        st.caption(f"Effective device: {format_compute_device_label(effective_compute_device)}")
        if compute_device_warning:
            st.warning(compute_device_warning)

        if _button_stretch("New Chat"):
            action["new_chat"] = True
        if _button_stretch("Clear current branch"):
            action["clear_chat"] = True

        st.markdown("### Branches", unsafe_allow_html=False)
        if branch_rows:
            branch_labels = [_branch_label(branch) for branch in branch_rows]
            selected_branch_label = st.selectbox(
                "Branch selector",
                options=branch_labels,
                index=branch_index,
            )
            selected_branch = branch_rows[branch_labels.index(selected_branch_label)]
            parent_branch_id = str(selected_branch.get("parent_branch_id", "") or "")
            if parent_branch_id:
                st.caption(f"Parent branch: {parent_branch_id}")
            if _button_stretch("Switch branch"):
                action["switch_branch_id"] = str(selected_branch.get("branch_id", ""))
        else:
            st.caption("Only the main branch exists.")

        st.markdown("### Export", unsafe_allow_html=False)
        if export_markdown:
            st.download_button(
                "Download branch (.md)",
                data=export_markdown,
                file_name="branch-export.md",
                mime="text/markdown",
                use_container_width=True,
            )
        if export_json:
            st.download_button(
                "Download branch (.json)",
                data=export_json,
                file_name="branch-export.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("### Cache Maintenance", unsafe_allow_html=False)
        if _button_stretch("Clear query cache"):
            action["clear_query_cache"] = True
        if _button_stretch("Clear answer cache"):
            action["clear_answer_cache"] = True
        if _button_stretch("Clear paper cache"):
            action["clear_paper_cache"] = True

        metrics = dict(last_response_metrics or {})
        if metrics:
            st.markdown("### Latency", unsafe_allow_html=False)
            for label, key in (
                ("Answer cache lookup", "answer_cache_lookup_ms"),
                ("Retrieval", "retrieval_ms"),
                ("Generation", "llm_ms"),
                ("Total", "total_ms"),
            ):
                if key not in metrics:
                    continue
                st.metric(label, f"{float(metrics[key]):.1f} ms")

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


def render_chat(
    messages: list[dict],
    *,
    top_n: int,
    show_papers: bool,
    show_rewritten_query: bool,
) -> dict[str, Any]:
    action = {"edit_message_index": None}
    for idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            edit_requested = render_message(
                message,
                top_n=top_n,
                show_papers=show_papers,
                show_rewritten_query=show_rewritten_query,
                message_key=f"message_{idx}",
                allow_prompt_edit=bool(message.get("role") == "user"),
            )
            if edit_requested:
                action["edit_message_index"] = idx
    return action


def render_message(
    message: Mapping[str, Any],
    *,
    top_n: int,
    show_papers: bool = True,
    show_rewritten_query: bool = False,
    message_key: str = "",
    allow_prompt_edit: bool = False,
) -> bool:
    role = message.get("role")
    content = strip_reframe_block(str(message.get("content", "") or ""))
    if role == "assistant":
        if bool(message.get("answer_cache_hit")):
            timestamp = _format_timestamp(str(message.get("answer_cache_created_at", "") or ""))
            similarity = float(message.get("answer_cache_similarity", 0.0) or 0.0)
            label = "Cached answer (similar query)"
            if timestamp:
                label = f"{label} · {timestamp}"
            if similarity:
                label = f"{label} · {similarity:.2f}"
            st.markdown(f'<div class="cache-pill">{label}</div>', unsafe_allow_html=True)
        cache_note = str(message.get("answer_cache_note", "") or "").strip()
        if cache_note:
            st.info(cache_note)

        st.markdown(beautify_text(content), unsafe_allow_html=False)
        rewritten_query = str(message.get("rewritten_query", "") or "").strip()
        if rewritten_query and show_rewritten_query:
            st.caption(f"Interpreting your question as: {rewritten_query}")
        source_count_note = str(message.get("source_count_note", "") or "").strip()
        if source_count_note:
            st.caption(source_count_note)
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
        return False

    st.markdown(content, unsafe_allow_html=False)
    if allow_prompt_edit and _button_stretch("Edit prompt", key=f"edit_{message_key}"):
        return True
    return False


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
    context = str(source.get("context", "") or "").strip()

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
    if context:
        with st.expander("Source inspector", expanded=False):
            st.markdown(beautify_text(context), unsafe_allow_html=False)


def classify_query_topic(query: str) -> str:
    normalized = str(query or "").lower()
    for label, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return label
    return "general"


def get_thinking_message(query: str) -> str:
    topic = classify_query_topic(query)
    messages = THINKING_MESSAGES.get(topic) or THINKING_MESSAGES["general"]
    if not messages:
        return "Thinking..."
    index = _stable_index(str(query or ""), len(messages))
    return messages[index]


def format_compute_device_label(device: str) -> str:
    normalized = str(device or "").strip().lower()
    if normalized == "cuda":
        return "GPU"
    if normalized == "cpu":
        return "CPU"
    return normalized.upper() or "AUTO"


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
        ranked_item: SourceItem = {
            "rank": len(ranked) + 1,
            "pmid": pmid,
            "title": str(item.get("title", "") or ""),
            "journal": str(item.get("journal", "") or ""),
            "year": str(item.get("year", "") or ""),
            "doi": str(item.get("doi", "") or ""),
            "pmcid": str(item.get("pmcid", "") or ""),
        }
        fulltext_url = str(item.get("fulltext_url", "") or "")
        if fulltext_url:
            ranked_item["fulltext_url"] = fulltext_url
        context = str(item.get("context", "") or "").strip()
        if context:
            ranked_item["context"] = context
        ranked.append(ranked_item)
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


def _branch_label(branch: Mapping[str, Any]) -> str:
    title = str(branch.get("title", "") or "Conversation branch")
    branch_id = str(branch.get("branch_id", "") or "")
    parent_branch_id = str(branch.get("parent_branch_id", "") or "")
    prefix = "* " if bool(branch.get("is_active")) else ""
    if parent_branch_id:
        return f"{prefix}{title} [{branch_id}] <- {parent_branch_id}"
    return f"{prefix}{title} [{branch_id}]"


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
                        const toast = document.createElement("div");
                        toast.innerText = "Copied";
                        toast.style.cssText = "position:fixed;bottom:16px;right:16px;background:#111;color:#fff;padding:8px 12px;border-radius:8px;font-size:12px;z-index:9999;opacity:0.95;";
                        window.parent.document.body.appendChild(toast);
                        setTimeout(() => toast.remove(), 1200);
                    }} catch (e) {{
                        this.innerText = "Copy failed";
                    }}
                }})()'
            >Copy response</button>
        </div>
        """,
        height=40,
    )


def _stable_index(text: str, size: int) -> int:
    if size <= 0:
        return 0
    return sum(ord(char) for char in str(text or "")) % size
