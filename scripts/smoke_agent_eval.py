from __future__ import annotations

import json
from pathlib import Path

from eval.evaluator import evaluate_turn
from eval.store import EvalStore
from src.agent.orchestrator import invoke_agent_chat
from src.core.config import load_config
from src.core.pipeline import invoke_chat


def main() -> None:
    config = load_config()
    query = "DOACs vs warfarin for stroke prevention in atrial fibrillation"
    session_id = "smoke-session"
    top_n = 5

    baseline = invoke_chat(query, session_id=session_id, top_n=top_n)
    agent = invoke_agent_chat(query, session_id=f"{session_id}-agent", top_n=top_n)

    eval_record = evaluate_turn(
        query=query,
        answer=str(agent.get("answer", "")),
        contexts=agent.get("retrieved_contexts", []) or [],
        sources=agent.get("sources", []) or [],
        mode="smoke",
    )

    store = EvalStore(config.eval_store_path)
    store.append(eval_record)

    output = {
        "baseline_status": baseline.get("status"),
        "agent_status": agent.get("status"),
        "baseline_sources": len(baseline.get("sources", []) or []),
        "agent_sources": len(agent.get("sources", []) or []),
        "eval_passed": eval_record.get("passed"),
        "eval_store": str(Path(config.eval_store_path).resolve()),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
