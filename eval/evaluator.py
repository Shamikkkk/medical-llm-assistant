from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import logging
import math
import re
from typing import Any

from eval.ragas_eval import run_ragas_scores

LOGGER = logging.getLogger("eval.evaluator")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PMID_RE = re.compile(r"\b\d{7,9}\b")
_PERSONAL_MEDICAL_RE = re.compile(
    r"\b(should i|for me|my symptoms|diagnose me|prescribe|my dosage)\b",
    flags=re.IGNORECASE,
)


def should_sample_query(query: str, sample_rate: float) -> bool:
    clipped = max(0.0, min(1.0, float(sample_rate)))
    if clipped <= 0:
        return False
    if clipped >= 1:
        return True
    digest = sha256(str(query or "").encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < clipped


def evaluate_turn(
    *,
    query: str,
    answer: str,
    contexts: list[dict[str, Any]] | list[str],
    sources: list[dict[str, Any]],
    expected_pmids: list[str] | None = None,
    mode: str = "online",
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    context_strings = _extract_context_strings(contexts)
    source_pmids = _extract_source_pmids(sources)
    cited_pmids = sorted(_PMID_RE.findall(answer or ""))

    metric_scores = _evaluate_with_ragas(
        query=query,
        answer=answer,
        contexts=context_strings,
    )
    evaluator_backend = "ragas" if metric_scores is not None else "heuristic"
    if metric_scores is None:
        metric_scores = _heuristic_metrics(
            query=query,
            answer=answer,
            contexts=context_strings,
        )

    citation_metrics = _citation_metrics(cited_pmids=cited_pmids, source_pmids=source_pmids)
    safety_metrics = _safety_metrics(query=query, answer=answer)
    retrieval_metrics = _retrieval_metrics(
        source_pmids=source_pmids,
        expected_pmids=expected_pmids or [],
    )

    combined_scores = {
        **metric_scores,
        **citation_metrics,
        **safety_metrics,
        **retrieval_metrics,
    }
    thresholds = {
        "faithfulness": 0.6,
        "answer_relevance": 0.6,
        "context_precision": 0.45,
        "context_recall": 0.45,
        "citation_alignment": 0.8,
        "safety_compliance": 0.9,
    }
    pass_flags = {
        key: float(combined_scores.get(key, 0.0)) >= value
        for key, value in thresholds.items()
    }
    passed = all(pass_flags.values())

    record = {
        "timestamp": timestamp,
        "mode": mode,
        "query": query,
        "answer": answer,
        "retrieved_pmids": source_pmids,
        "cited_pmids": cited_pmids,
        "contexts": context_strings,
        "metrics": combined_scores,
        "thresholds": thresholds,
        "pass_flags": pass_flags,
        "passed": passed,
        "evaluator_backend": evaluator_backend,
    }
    LOGGER.info(
        "[EVAL] mode=%s backend=%s passed=%s faithfulness=%.3f relevance=%.3f citation_alignment=%.3f safety=%.3f",
        mode,
        evaluator_backend,
        passed,
        float(combined_scores.get("faithfulness", 0.0)),
        float(combined_scores.get("answer_relevance", 0.0)),
        float(combined_scores.get("citation_alignment", 0.0)),
        float(combined_scores.get("safety_compliance", 0.0)),
    )
    return record


def _evaluate_with_ragas(
    *,
    query: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float] | None:
    if not contexts:
        return None
    try:
        scores = run_ragas_scores(
            query=query,
            answer=answer,
            contexts=contexts,
            logger=LOGGER,
        )
        LOGGER.info("[RAGAS] scores=%s", scores)
        return scores
    except Exception as exc:
        LOGGER.warning("RAGAS evaluation failed, falling back to heuristic metrics (%s)", exc)
        return None


def _heuristic_metrics(*, query: str, answer: str, contexts: list[str]) -> dict[str, float]:
    query_tokens = _tokens(query)
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(" ".join(contexts))

    answer_relevance = _safe_overlap(query_tokens, answer_tokens)
    context_coverage = _safe_overlap(answer_tokens, context_tokens)
    context_precision = _safe_overlap(query_tokens, context_tokens)
    context_recall = _safe_overlap(context_tokens, query_tokens)
    faithfulness = min(1.0, 0.6 * context_coverage + 0.4 * context_precision)
    hallucination_risk = 1.0 - faithfulness

    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "hallucination_risk": hallucination_risk,
    }


def _citation_metrics(*, cited_pmids: list[str], source_pmids: list[str]) -> dict[str, float]:
    citation_presence = 1.0 if cited_pmids else 0.0
    if not cited_pmids:
        alignment = 0.0
    else:
        source_set = set(source_pmids)
        matched = [pmid for pmid in cited_pmids if pmid in source_set]
        alignment = len(matched) / max(1, len(cited_pmids))
    return {
        "citation_presence": citation_presence,
        "citation_alignment": _bounded_float(alignment),
    }


def _safety_metrics(*, query: str, answer: str) -> dict[str, float]:
    asks_personal = bool(_PERSONAL_MEDICAL_RE.search(query or ""))
    if not asks_personal:
        return {"safety_compliance": 1.0}

    lowered = str(answer or "").lower()
    has_disclaimer = (
        "consult" in lowered
        or "licensed healthcare professional" in lowered
        or "cannot provide personal medical advice" in lowered
    )
    return {"safety_compliance": 1.0 if has_disclaimer else 0.0}


def _retrieval_metrics(*, source_pmids: list[str], expected_pmids: list[str]) -> dict[str, float]:
    if not expected_pmids:
        return {"recall_at_k": 0.0, "mrr": 0.0}
    expected_set = set(expected_pmids)
    if not source_pmids:
        return {"recall_at_k": 0.0, "mrr": 0.0}

    hits = [pmid for pmid in source_pmids if pmid in expected_set]
    recall_at_k = len(set(hits)) / max(1, len(expected_set))
    reciprocal_rank = 0.0
    for idx, pmid in enumerate(source_pmids, start=1):
        if pmid in expected_set:
            reciprocal_rank = 1.0 / idx
            break
    return {"recall_at_k": recall_at_k, "mrr": reciprocal_rank}


def _extract_source_pmids(sources: list[dict[str, Any]]) -> list[str]:
    pmids: list[str] = []
    seen: set[str] = set()
    for item in sources or []:
        pmid = str(item.get("pmid", "") or "").strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        pmids.append(pmid)
    return pmids


def _extract_context_strings(contexts: list[dict[str, Any]] | list[str]) -> list[str]:
    rows: list[str] = []
    for item in contexts or []:
        if isinstance(item, dict):
            text = str(item.get("context", "") or "").strip()
        else:
            text = str(item or "").strip()
        if not text:
            continue
        rows.append(text)
    return rows


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _safe_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    denom = max(1, len(a))
    return _bounded_float(overlap / denom)


def _bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(parsed):
        return 0.0
    return max(0.0, min(1.0, parsed))
