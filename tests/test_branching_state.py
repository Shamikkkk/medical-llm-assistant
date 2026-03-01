from __future__ import annotations

from unittest import TestCase

from src.app_state import build_branched_messages
from src.history import clear_session_history, get_session_history, replace_session_history


class BranchingStateTests(TestCase):
    def tearDown(self) -> None:
        clear_session_history("chat-1")

    def test_build_branched_messages_replaces_target_and_truncates_tail(self) -> None:
        messages = [
            {"role": "user", "content": "Original question"},
            {"role": "assistant", "content": "Original answer"},
            {"role": "user", "content": "Follow-up question"},
            {"role": "assistant", "content": "Follow-up answer"},
        ]

        branched = build_branched_messages(
            messages,
            message_index=2,
            edited_content="Edited follow-up question",
        )

        self.assertEqual(
            branched,
            [
                {"role": "user", "content": "Original question"},
                {"role": "assistant", "content": "Original answer"},
                {"role": "user", "content": "Edited follow-up question"},
            ],
        )

    def test_histories_are_separated_by_branch_id(self) -> None:
        replace_session_history(
            "chat-1",
            "main",
            [
                {"role": "user", "content": "Main branch question"},
                {"role": "assistant", "content": "Main branch answer"},
            ],
        )
        replace_session_history(
            "chat-1",
            "branch-a",
            [
                {"role": "user", "content": "Branch question"},
            ],
        )

        main_history = get_session_history("chat-1", "main")
        branch_history = get_session_history("chat-1", "branch-a")

        self.assertIsNot(main_history, branch_history)
        self.assertEqual(len(main_history.messages), 2)
        self.assertEqual(len(branch_history.messages), 1)
