from __future__ import annotations

import json
from unittest import TestCase

from src.utils.export import export_branch_json, export_branch_markdown


class ExportHelperTests(TestCase):
    def test_export_branch_markdown_includes_branch_metadata(self) -> None:
        markdown = export_branch_markdown(
            chat_title="Clinical chat",
            branch_title="Edited branch",
            branch_id="branch-1",
            parent_branch_id="main",
            messages=[
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        )

        self.assertIn("# Clinical chat", markdown)
        self.assertIn("`branch-1`", markdown)
        self.assertIn("Parent branch", markdown)
        self.assertIn("## User", markdown)
        self.assertIn("## Assistant", markdown)

    def test_export_branch_json_serializes_messages(self) -> None:
        payload = export_branch_json(
            chat_id="chat-1",
            chat_title="Clinical chat",
            branch={
                "branch_id": "branch-1",
                "title": "Edited branch",
                "parent_branch_id": "main",
                "parent_turn_index": 2,
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            messages=[{"role": "user", "content": "Question"}],
        )

        parsed = json.loads(payload)
        self.assertEqual(parsed["chat_id"], "chat-1")
        self.assertEqual(parsed["branch"]["branch_id"], "branch-1")
        self.assertEqual(parsed["messages"][0]["content"], "Question")
