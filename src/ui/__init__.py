"""Streamlit rendering helpers."""

from src.ui.formatters import beautify_text, doi_url, pubmed_url, strip_reframe_block
from src.ui.loading_messages import detect_topic, pick_loading_message

__all__ = [
    "beautify_text",
    "detect_topic",
    "doi_url",
    "pick_loading_message",
    "pubmed_url",
    "strip_reframe_block",
    "auto_scroll",
    "apply_app_styles",
    "build_auto_scroll_html",
    "classify_query_topic",
    "get_thinking_message",
    "render_chat",
    "render_header",
    "render_message",
    "render_ranked_sources",
    "render_sidebar",
    "render_source_item",
    "render_sources",
]

try:  # pragma: no cover - optional during lightweight test environments
    from src.ui.render import (
        auto_scroll,
        apply_app_styles,
        build_auto_scroll_html,
        classify_query_topic,
        get_thinking_message,
        render_chat,
        render_header,
        render_message,
        render_ranked_sources,
        render_sidebar,
        render_source_item,
        render_sources,
    )
except Exception:  # pragma: no cover
    pass
