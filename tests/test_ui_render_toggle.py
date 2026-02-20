from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless

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
