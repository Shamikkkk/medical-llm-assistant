from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch


class ConfigLoadingTests(TestCase):
    def test_legacy_max_abstracts_is_treated_as_context_budget_alias(self) -> None:
        from src.core.config import load_config

        with patch.dict(
            os.environ,
            {
                "MAX_ABSTRACTS": "7",
            },
            clear=False,
        ):
            os.environ.pop("MAX_CONTEXT_ABSTRACTS", None)
            config = load_config()

        self.assertEqual(config.max_context_abstracts, 7)
        self.assertEqual(config.max_abstracts, 7)
        self.assertTrue(
            any("MAX_ABSTRACTS is deprecated" in warning for warning in config.config_warnings)
        )

    def test_max_context_abstracts_takes_precedence_over_legacy_alias(self) -> None:
        from src.core.config import load_config

        with patch.dict(
            os.environ,
            {
                "MAX_ABSTRACTS": "6",
                "MAX_CONTEXT_ABSTRACTS": "9",
            },
            clear=False,
        ):
            config = load_config()

        self.assertEqual(config.max_context_abstracts, 9)
        self.assertEqual(config.max_abstracts, 9)
        self.assertTrue(
            any("Using MAX_CONTEXT_ABSTRACTS" in warning for warning in config.config_warnings)
        )
