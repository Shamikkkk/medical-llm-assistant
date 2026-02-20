from __future__ import annotations

import os
from typing import Any


def run_ragas_scores(
    *,
    query: str,
    answer: str,
    contexts: list[str],
    logger: Any,
) -> dict[str, float]:
    """Run RAGAS using an OpenAI-compatible endpoint (NVIDIA NIM)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        pass

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    judge_model = (
        os.getenv("RAGAS_JUDGE_MODEL", "").strip()
        or os.getenv("NVIDIA_MODEL", "").strip()
        or "meta/llama-3.1-8b-instruct"
    )

    logger.info(
        "[EVAL] RAGAS judge configured: base_url=%s, api_key=%s",
        openai_base_url or "NOT_SET",
        "SET" if bool(openai_api_key) else "NOT_SET",
    )

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set for RAGAS judge.")
    if not openai_base_url:
        raise ValueError("OPENAI_BASE_URL is not set for RAGAS judge.")

    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge_llm = ChatOpenAI(
        model=judge_model,
        temperature=0,
        api_key=openai_api_key,
        base_url=openai_base_url,
    )
    wrapped_llm = LangchainLLMWrapper(judge_llm)

    dataset = Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [""],
        }
    )
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=wrapped_llm,
    )
    data = result.to_pandas().to_dict("records")[0]
    return {
        "faithfulness": _bounded_float(data.get("faithfulness")),
        "answer_relevance": _bounded_float(data.get("answer_relevancy")),
        "context_precision": _bounded_float(data.get("context_precision")),
        "context_recall": _bounded_float(data.get("context_recall")),
        "hallucination_risk": 1.0 - _bounded_float(data.get("faithfulness")),
    }


def _bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))
