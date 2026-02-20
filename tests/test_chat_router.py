from __future__ import annotations

import importlib.util
from unittest import TestCase, skipUnless
from unittest.mock import patch

LANGCHAIN_NVIDIA_AVAILABLE = (
    importlib.util.find_spec("langchain_nvidia_ai_endpoints") is not None
)


@skipUnless(LANGCHAIN_NVIDIA_AVAILABLE, "langchain_nvidia_ai_endpoints is not installed")
class ChatRouterTests(TestCase):
    @patch("src.chat.router.invoke_chat_with_mode")
    @patch("src.chat.router._get_llm_safe")
    def test_router_removes_legacy_paper_payload_fields(
        self,
        mock_llm_safe,
        mock_invoke_chat_with_mode,
    ) -> None:
        from src.chat.router import invoke_chat_request

        mock_llm_safe.return_value = None
        mock_invoke_chat_with_mode.return_value = {
            "status": "answered",
            "answer": "ok",
            "paper_focus_mode": True,
            "paper_focus_notice": "legacy",
        }
        payload = invoke_chat_request(
            query="DOACs vs warfarin",
            session_id="s1",
            top_n=5,
            agent_mode=False,
            follow_up_mode=False,
            chat_messages=[],
        )
        self.assertNotIn("paper_focus_mode", payload)
        self.assertNotIn("paper_focus_notice", payload)
        self.assertEqual(payload.get("query"), "DOACs vs warfarin")

    @patch("src.chat.router.invoke_chat_with_mode")
    @patch("src.chat.router._get_llm_safe")
    def test_router_uses_rewritten_query_for_pipeline_call(
        self,
        mock_llm_safe,
        mock_invoke_chat_with_mode,
    ) -> None:
        from src.chat.router import invoke_chat_request

        class _Response:
            content = "Standalone query for HFpEF treatment evidence"

        class _Llm:
            def invoke(self, _: str):
                return _Response()

        mock_llm_safe.return_value = _Llm()
        mock_invoke_chat_with_mode.return_value = {"status": "answered", "answer": "ok"}
        invoke_chat_request(
            query="What about that in HFpEF?",
            session_id="s2",
            top_n=5,
            agent_mode=False,
            follow_up_mode=True,
            chat_messages=[
                {"role": "user", "content": "Tell me about heart failure treatment."},
                {"role": "assistant", "content": "Discussed HFrEF therapy evidence."},
            ],
        )
        called_query = mock_invoke_chat_with_mode.call_args.args[0]
        self.assertIn("HFpEF", called_query)
