"""Streamlit rendering helpers."""

from src.ui.formatters import beautify_text, doi_url, pubmed_url, strip_reframe_block
from src.ui.loading_messages import detect_topic, pick_loading_message

try:  # pragma: no cover - optional during lightweight test environments
    from src.ui.render import (
        apply_app_styles,
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

__all__ = [
    "beautify_text",
    "detect_topic",
    "doi_url",
    "pick_loading_message",
    "pubmed_url",
    "strip_reframe_block",
]

for _name in (
    "apply_app_styles",
    "render_chat",
    "render_header",
    "render_message",
    "render_ranked_sources",
    "render_sidebar",
    "render_source_item",
    "render_sources",
):
    if _name in globals():
        __all__.append(_name)
