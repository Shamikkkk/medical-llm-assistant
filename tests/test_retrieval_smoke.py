from __future__ import annotations

import os
from unittest import TestCase


class RetrievalSmokeTests(TestCase):
    def test_retrieval_non_empty_for_known_query(self) -> None:
        if os.getenv("RUN_NETWORK_TESTS", "false").lower() not in {"1", "true", "yes"}:
            self.skipTest("Set RUN_NETWORK_TESTS=true to run network retrieval smoke test.")

        from src.core.pipeline import invoke_chat

        payload = invoke_chat(
            "DOACs vs warfarin for stroke prevention in atrial fibrillation",
            session_id="retrieval-smoke",
            top_n=3,
        )
        self.assertIn(payload.get("status"), {"answered", "cache_miss", "cache_hit"})
        sources = payload.get("sources", []) or []
        self.assertGreater(len(sources), 0, "Expected at least one retrieved source.")
