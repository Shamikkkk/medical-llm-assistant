from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless

LANGCHAIN_HISTORY_AVAILABLE = (
    importlib.util.find_spec("langchain_core") is not None
    and importlib.util.find_spec("langchain_community") is not None
)


@skipUnless(
    LANGCHAIN_HISTORY_AVAILABLE,
    "langchain_core/langchain_community is not installed",
)
class ScopeDomainAgnosticTests(TestCase):
    def test_oncology_query_is_in_scope(self) -> None:
        from src.core.scope import classify_scope

        result = classify_scope(
            "Glioblastoma temozolomide survival evidence in newly diagnosed adults",
            session_id="scope-oncology",
            llm=None,
        )
        self.assertTrue(result.allow)
        self.assertIn(result.label, {"BIOMEDICAL", "MULTI_SYSTEM_OVERLAP"})

    def test_gi_query_is_in_scope(self) -> None:
        from src.core.scope import classify_scope

        result = classify_scope(
            "IBS low FODMAP diet evidence for symptom reduction",
            session_id="scope-gi",
            llm=None,
        )
        self.assertTrue(result.allow)
        self.assertIn(result.label, {"BIOMEDICAL", "MULTI_SYSTEM_OVERLAP"})

    def test_non_biomedical_query_is_out_of_scope(self) -> None:
        from src.core.scope import classify_scope

        result = classify_scope(
            "Who will win the football league this season?",
            session_id="scope-sports",
            llm=None,
        )
        self.assertFalse(result.allow)
        self.assertEqual(result.label, "OUT_OF_SCOPE")
