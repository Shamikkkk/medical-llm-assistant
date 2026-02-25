from __future__ import annotations

from unittest import TestCase

from src.ui.loading_messages import detect_topic, pick_loading_message


class LoadingMessageTests(TestCase):
    def test_detect_topic_gi(self) -> None:
        topic = detect_topic("What is the evidence for low FODMAP diet in IBS?")
        self.assertEqual(topic, "gi")
        self.assertEqual(pick_loading_message(topic, "query"), "Digesting the literature...")

    def test_detect_topic_oncology(self) -> None:
        topic = detect_topic("Temozolomide in glioblastoma after chemoradiation")
        self.assertEqual(topic, "oncology")
        self.assertEqual(
            pick_loading_message(topic, "query"),
            "Scanning tumor biology and trial evidence...",
        )

    def test_detect_topic_general_fallback(self) -> None:
        topic = detect_topic("Can you summarize this?")
        self.assertEqual(topic, "general")
        self.assertEqual(pick_loading_message(topic, "query"), "Thinking...")
