from __future__ import annotations

from unittest import TestCase

from src.chat.contextualize import contextualize_question


class _DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyLlm:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_prompt = ""

    def invoke(self, prompt: str):
        self.last_prompt = prompt
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

    def test_contextualize_uses_conversation_summary_and_last_question(self) -> None:
        llm = _DummyLlm("Standalone question about DOAC bleeding risk in atrial fibrillation")
        messages = [
            {"role": "user", "content": "Summarize DOAC versus warfarin evidence in AF."},
            {"role": "assistant", "content": "We reviewed stroke and bleeding outcomes."},
        ]
        rewritten, summary, used = contextualize_question(
            user_query="What about bleeding risk?",
            chat_messages=messages,
            follow_up_mode=True,
            conversation_summary="Conversation summary: DOACs in AF with focus on safety outcomes.",
            llm=llm,
        )
        self.assertTrue(used)
        self.assertIn("DOAC", rewritten)
        self.assertIn("Conversation summary", llm.last_prompt)
        self.assertIn("Summarize DOAC versus warfarin evidence in AF.", llm.last_prompt)
        self.assertIn("DOACs in AF", summary)
