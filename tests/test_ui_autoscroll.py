from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import patch

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None


@skipUnless(STREAMLIT_AVAILABLE, "streamlit is not installed")
class UiAutoScrollTests(TestCase):
    @patch("src.ui.render.components.html")
    def test_auto_scroll_injects_smooth_scroll_script(self, mock_html) -> None:
        from src.ui.render import auto_scroll

        auto_scroll()

        mock_html.assert_called_once()
        args, kwargs = mock_html.call_args
        script = str(args[0])
        self.assertIn("scrollIntoView", script)
        self.assertIn("smooth", script)
        self.assertEqual(kwargs.get("height"), 0)

    def test_build_auto_scroll_html_disabled_returns_marker(self) -> None:
        from src.ui.render import build_auto_scroll_html

        html = build_auto_scroll_html(enabled=False)

        self.assertIn("chat-autoscroll-disabled", html)
