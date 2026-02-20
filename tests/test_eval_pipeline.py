from __future__ import annotations

from unittest import TestCase

from eval.evaluator import evaluate_turn


class EvaluationPipelineTests(TestCase):
    def test_evaluation_runs_end_to_end(self) -> None:
        record = evaluate_turn(
            query="DOACs vs warfarin for AF",
            answer="DOACs may reduce stroke risk compared with warfarin [41166960].",
            contexts=[
                {
                    "pmid": "41166960",
                    "context": "In AF populations, DOACs were associated with lower stroke and bleeding risk.",
                }
            ],
            sources=[{"pmid": "41166960", "title": "Example", "rank": 1}],
            mode="test",
        )
        self.assertIn("metrics", record)
        self.assertIn("faithfulness", record["metrics"])
        self.assertIn("passed", record)
