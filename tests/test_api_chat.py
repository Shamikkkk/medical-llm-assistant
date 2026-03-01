# Created by Codex - Section 1

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.dependencies import get_session_store
from api.main import app
from api.session_store import SessionStore


class ApiChatTests(TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.store = SessionStore(Path(self.temp_dir.name) / "sessions.json")
        app.dependency_overrides[get_session_store] = lambda: self.store

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.temp_dir.cleanup()

    @patch("api.routers.chat.invoke_chat_request")
    def test_invoke_endpoint_returns_payload_and_persists_messages(self, mock_invoke) -> None:
        mock_invoke.return_value = {
            "status": "answered",
            "answer": "Assistant reply",
            "sources": [{"pmid": "12345678"}],
            "timings": {"total_ms": 10.0},
        }

        client = TestClient(app)
        response = client.post(
            "/api/chat/invoke",
            json={
                "query": "What is the evidence?",
                "session_id": "chat-1",
                "branch_id": "main",
                "top_n": 5,
                "agent_mode": False,
                "follow_up_mode": True,
                "chat_messages": [],
                "show_papers": True,
                "conversation_summary": "",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()["payload"]
        self.assertEqual(payload["answer"], "Assistant reply")
        messages = self.store.get_messages("chat-1", "main")
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])

    @patch("api.routers.chat.stream_chat_request")
    def test_stream_endpoint_emits_chunk_and_done_events(self, mock_stream) -> None:
        def _generator():
            yield "Partial "
            yield "answer"
            return {"status": "answered", "answer": "Partial answer", "sources": []}

        mock_stream.return_value = _generator()

        client = TestClient(app)
        response = client.post(
            "/api/chat/stream",
            json={
                "query": "Tell me more",
                "session_id": "chat-2",
                "branch_id": "main",
                "top_n": 5,
                "agent_mode": False,
                "follow_up_mode": True,
                "chat_messages": [],
                "show_papers": True,
                "conversation_summary": "",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "chunk"', response.text)
        self.assertIn('"type": "done"', response.text)
        messages = self.store.get_messages("chat-2", "main")
        self.assertEqual(messages[-1]["content"], "Partial answer")
