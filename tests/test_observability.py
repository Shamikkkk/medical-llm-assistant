from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import Mock, patch

from src.logging_utils import log_event
from src.observability.metrics import summarize_metric_events
from src.observability.tracing import start_span


class ObservabilityTests(TestCase):
    def test_log_event_emits_valid_json(self) -> None:
        logger = Mock()
        with patch("src.logging_utils.get_logger", return_value=logger):
            payload = log_event(
                "request.complete",
                request_id="req-1",
                session_id="session-1",
                total_ms=12.3,
            )

        logger.info.assert_called_once()
        logged_json = logger.info.call_args.args[0]
        parsed = json.loads(logged_json)
        self.assertEqual(parsed["event"], "request.complete")
        self.assertEqual(parsed["request_id"], "req-1")
        self.assertEqual(payload["session_id"], "session-1")

    def test_start_span_is_noop_without_tracer(self) -> None:
        with patch("src.observability.tracing._get_tracer", return_value=None):
            with start_span("llm.generate") as span:
                self.assertIsNone(span)

    def test_metrics_summary_aggregates_request_events(self) -> None:
        summary = summarize_metric_events(
            [
                {
                    "event": "request.complete",
                    "total_ms": 100.0,
                    "cache_hit": True,
                    "pmid_count": 3,
                },
                {
                    "event": "request.complete",
                    "total_ms": 300.0,
                    "cache_hit": False,
                    "pmid_count": 1,
                },
                {
                    "event": "request.error",
                    "total_ms": 50.0,
                },
            ]
        )

        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["error_count"], 1)
        self.assertAlmostEqual(summary["error_rate"], 1 / 3)
        self.assertEqual(summary["latency_p50_ms"], 200.0)
        self.assertEqual(summary["cache_hit_rate"], 0.5)
        self.assertEqual(summary["avg_pmid_count"], 2.0)
