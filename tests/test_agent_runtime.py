from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from src.agent.runtime import invoke_chat_with_mode


class AgentRuntimeTests(TestCase):
    def test_non_agent_mode_uses_baseline_pipeline(self) -> None:
        expected = {"status": "answered", "answer": "baseline"}
        with patch("src.agent.runtime._invoke_pipeline", return_value=expected) as mocked_baseline:
            result = invoke_chat_with_mode(
                "test query",
                session_id="s1",
                top_n=5,
                agent_mode=False,
            )
        self.assertEqual(result, expected)
        mocked_baseline.assert_called_once_with("test query", session_id="s1", top_n=5)

    def test_agent_mode_uses_orchestrator(self) -> None:
        expected = {"status": "answered", "answer": "agent"}
        with patch("src.agent.runtime._invoke_agent", return_value=expected) as mocked_agent:
            result = invoke_chat_with_mode(
                "test query",
                session_id="s1",
                top_n=5,
                agent_mode=True,
            )
        self.assertEqual(result, expected)
        mocked_agent.assert_called_once_with("test query", session_id="s1", top_n=5)
