from __future__ import annotations

from unittest import TestCase

from src.chat.contextualize import contextualize_question


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyLlm:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _: str):
        return _DummyResponse(self._content)


class FollowupContextualizationTests(TestCase):
    def test_followup_mode_rewrites_even_when_query_is_long(self) -> None:
        llm = _DummyLlm("Standalone biomedical query about side effects of SGLT2 inhibitors")
        messages = [
            {"role": "user", "content": "Tell me about SGLT2 inhibitors in heart failure."},
            {"role": "assistant", "content": "They reduce HF hospitalization in key trials."},
        ]
        rewritten, _, used = contextualize_question(
            user_query="Can you compare adverse event patterns across these studies in detail?",
            chat_messages=messages,
            follow_up_mode=True,
            llm=llm,
        )
        self.assertTrue(used)
        self.assertIn("SGLT2", rewritten)

    def test_followup_like_query_rewrites_even_if_toggle_off(self) -> None:
        messages = [
            {"role": "user", "content": "Summarize DOAC versus warfarin evidence in AF."},
            {"role": "assistant", "content": "DOACs reduced several major outcomes."},
        ]
        rewritten, _, used = contextualize_question(
            user_query="What about side effects?",
            chat_messages=messages,
            follow_up_mode=False,
            llm=None,
        )
        self.assertTrue(used)
        self.assertNotEqual(rewritten, "What about side effects?")
