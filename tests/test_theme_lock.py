from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


class ThemeLockTests(TestCase):
    def test_streamlit_config_locks_dark_theme(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"

        self.assertTrue(config_path.exists())
        content = config_path.read_text(encoding="utf-8")
        self.assertIn('base = "dark"', content)
        self.assertIn('toolbarMode = "minimal"', content)

    @patch("src.ui.render.st.markdown")
    def test_app_styles_hide_main_menu(self, mock_markdown) -> None:
        from src.ui.render import apply_app_styles

        apply_app_styles()

        css = mock_markdown.call_args.args[0]
        self.assertIn("#MainMenu", css)
        self.assertIn('button[title="Main menu"]', css)
