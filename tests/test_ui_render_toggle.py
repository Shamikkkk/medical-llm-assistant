from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import patch

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@skipUnless(STREAMLIT_AVAILABLE, "streamlit is not installed")
class UiToggleTests(TestCase):
    def test_sources_hidden_when_show_papers_off(self) -> None:
        from src.ui.render import should_render_sources

        self.assertFalse(should_render_sources(status="answered", show_papers=False))

    def test_sources_shown_when_answered_and_show_papers_on(self) -> None:
        from src.ui.render import should_render_sources

        self.assertTrue(should_render_sources(status="answered", show_papers=True))

    def test_sources_hidden_for_non_answer_status(self) -> None:
        from src.ui.render import should_render_sources

        self.assertFalse(should_render_sources(status="out_of_scope", show_papers=True))

    def test_topic_classifier_maps_oncology_queries(self) -> None:
        from src.ui.render import classify_query_topic

        self.assertEqual(
            classify_query_topic("Temozolomide evidence for glioblastoma"),
            "oncology",
        )

    @patch("src.ui.render.components.html")
    def test_copy_button_renders_button_html(self, mock_html) -> None:
        from src.ui.render import _render_copy_button

        _render_copy_button("Answer text", key="copy-1")

        mock_html.assert_called_once()
        self.assertIn("Copy response", mock_html.call_args.args[0])

    def test_sidebar_exposes_new_toggles(self) -> None:
        from src.ui.render import render_sidebar

        toggle_labels: list[str] = []

        def _toggle(label: str, **_kwargs):
            toggle_labels.append(label)
            return False

        with (
            patch("src.ui.render.st.sidebar", new=nullcontext()),
            patch("src.ui.render.st.markdown"),
            patch("src.ui.render.st.caption"),
            patch("src.ui.render.st.slider", return_value=5),
            patch("src.ui.render.st.toggle", side_effect=_toggle),
            patch("src.ui.render.st.button", return_value=False),
        ):
            render_sidebar(
                chats=[],
                active_chat_id="chat-1",
                top_n=5,
                follow_up_mode=True,
                show_papers=False,
                show_rewritten_query=False,
                auto_scroll_enabled=True,
            )

        self.assertIn("Show papers", toggle_labels)
        self.assertIn("Show rewritten query", toggle_labels)
        self.assertIn("Auto-scroll", toggle_labels)
