from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.datasets import load_eval_dataset
from eval.evaluator import evaluate_turn
from eval.store import EvalStore
from src.agent.orchestrator import invoke_agent_chat
from src.core.config import load_config
from src.core.pipeline import invoke_chat


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation.")
    parser.add_argument("--dataset", required=True, help="Path to JSON/CSV evaluation dataset.")
    parser.add_argument(
        "--out",
        default="",
        help="Path to output JSONL. Defaults to EVAL_STORE_PATH from .env.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top-N papers per query.")
    parser.add_argument(
        "--agent-mode",
        action="store_true",
        help="Run with agent orchestrator instead of baseline pipeline.",
    )
    args = parser.parse_args()

    config = load_config()
    dataset = load_eval_dataset(args.dataset)
    output_path = args.out or str(config.eval_store_path)
    store = EvalStore(output_path)

    for idx, example in enumerate(dataset):
        query = str(example.get("query") or example.get("question") or "").strip()
        if not query:
            continue
        session_id = f"offline-{idx}"
        if args.agent_mode:
            payload = invoke_agent_chat(query, session_id=session_id, top_n=args.top_n)
            pipeline_mode = "agent"
        else:
            payload = invoke_chat(query, session_id=session_id, top_n=args.top_n)
            pipeline_mode = "baseline"

        answer = str(payload.get("answer") or payload.get("message") or "")
        contexts = payload.get("retrieved_contexts", []) or []
        expected_pmids = _coerce_expected_pmids(example.get("expected_pmids"))
        record = evaluate_turn(
            query=query,
            answer=answer,
            contexts=contexts,
            sources=payload.get("sources", []) or [],
            expected_pmids=expected_pmids,
            mode="offline",
        )
        record["pipeline_mode"] = pipeline_mode
        record["status"] = str(payload.get("status", "answered"))
        record["dataset_row"] = idx
        store.append(record)

    print(json.dumps({"rows_evaluated": len(dataset), "output": str(Path(output_path).resolve())}, indent=2))


def _coerce_expected_pmids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    main()
