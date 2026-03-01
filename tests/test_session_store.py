# Created by Codex - Section 1

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from api.session_store import MAIN_BRANCH_ID, SessionStore


class SessionStoreTests(TestCase):
    def test_create_chat_and_branch_persist_to_disk(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "sessions.json"
            store = SessionStore(store_path)

            chat = store.create_chat("Clinical review")
            store.append_message(chat["chat_id"], MAIN_BRANCH_ID, {"role": "user", "content": "Question"})
            store.append_message(chat["chat_id"], MAIN_BRANCH_ID, {"role": "assistant", "content": "Answer"})
            branch = store.create_branch(
                chat["chat_id"],
                MAIN_BRANCH_ID,
                0,
                "Edited question",
            )

            reloaded = SessionStore(store_path)
            chats = reloaded.get_chats()
            branches = reloaded.get_branches(chat["chat_id"])
            messages = reloaded.get_messages(chat["chat_id"], branch["branch_id"])

            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0]["title"], "Clinical review")
            self.assertEqual(len(branches), 2)
            self.assertEqual(messages[-1]["content"], "Edited question")

    def test_append_message_sets_chat_title_from_first_user_message(self) -> None:
        with TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir) / "sessions.json")
            store.append_message("chat-1", MAIN_BRANCH_ID, {"role": "user", "content": "What is new in HFpEF?"})

            chats = store.get_chats()
            self.assertEqual(chats[0]["chat_id"], "chat-1")
            self.assertIn("HFpEF", chats[0]["title"])
