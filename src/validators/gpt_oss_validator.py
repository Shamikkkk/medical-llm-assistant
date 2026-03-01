from __future__ import annotations

import logging
import re
from typing import Any

from src.validators.claim_splitter import split_into_claims
from src.validators.model_loader import DEFAULT_NLI_MODEL, get_nli_components
from src.validators.premise_builder import build_claim_premise
from src.validators.scoring import aggregate_claim_scores, score_claim_with_nli

LOGGER = logging.getLogger("validator.support")


def validate_answer(
    user_query: str,
    answer: str,
    context: str,
    source_pmids: list[str],
    *,
    model_name: str = DEFAULT_NLI_MODEL,
    threshold: float = 0.7,
    margin: float = 0.2,
    max_premise_tokens: int = 384,
    max_hypothesis_tokens: int = 128,
    max_length: int = 512,
    top_n_chunks: int = 4,
    top_k_sentences: int = 2,
    retrieved_docs: list[Any] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    del user_query  # reserved for future policy rules
    normalized_context = str(context or "").strip()
    normalized_answer = str(answer or "").strip()
    clipped_threshold = _clip(threshold, lower=0.0, upper=1.0, default=0.7)
    clipped_margin = _clip(margin, lower=0.0, upper=1.0, default=0.2)
    clipped_contradiction_limit = 0.20
    max_premise_tokens = int(_clip(max_premise_tokens, lower=64, upper=1024, default=384))
    max_hypothesis_tokens = int(_clip(max_hypothesis_tokens, lower=32, upper=512, default=128))
    max_length = int(_clip(max_length, lower=128, upper=1024, default=512))
    top_n_chunks = int(_clip(top_n_chunks, lower=1, upper=10, default=4))
    top_k_sentences = int(_clip(top_k_sentences, lower=1, upper=5, default=2))

    if not normalized_answer:
        return {
            "valid": True,
            "score": 0.0,
            "label": "NO_ANSWER",
            "details": {
                "reason": "Empty answer; validator skipped.",
                "model_name": model_name,
            },
        }

    evidence_chunks = _extract_evidence_chunks(retrieved_docs, normalized_context)
    if not evidence_chunks:
        return {
            "valid": True,
            "score": 0.0,
            "label": "NO_EVIDENCE",
            "details": {
                "reason": "No retrieved evidence provided to validator.",
                "model_name": model_name,
            },
        }

    claims = split_into_claims(normalized_answer)
    if not claims:
        claims = [normalized_answer]

    missing_pmids = sorted(_extract_pmids(normalized_answer) - {str(p).strip() for p in source_pmids if str(p).strip()})
    validator = get_nli_components(model_name, device=device)
    if validator is None:
        LOGGER.warning("Validator fallback: disabled because model failed to load.")
        return {
            "valid": True,
            "score": 0.0,
            "label": "VALIDATOR_DISABLED",
            "details": {
                "reason": "Validator unavailable; continuing without blocking.",
                "model_name": model_name,
            },
        }

    tokenizer = validator["tokenizer"]
    model = validator["model"]
    effective_model = str(validator.get("model_name", model_name))
    entailment_ready = bool(validator.get("entailment_ready", False))
    label_map = validator.get("label_map", {}) or {}
    runtime_heuristic_fallback = False

    claim_scores: list[dict[str, Any]] = []
    total_premise_tokens = 0
    total_hypothesis_tokens = 0
    truncation_flag = False

    for claim in claims:
        premise_info = build_claim_premise(
            claim=claim,
            evidence_chunks=evidence_chunks,
            tokenizer=tokenizer,
            top_n_chunks=top_n_chunks,
            top_k_sentences=top_k_sentences,
            max_premise_tokens=max_premise_tokens,
        )
        premise = str(premise_info.get("premise", "")).strip()
        if not premise:
            continue

        if entailment_ready:
            try:
                score_payload = score_claim_with_nli(
                    claim=claim,
                    premise=premise,
                    tokenizer=tokenizer,
                    model=model,
                    label_map=label_map,
                    max_premise_tokens=max_premise_tokens,
                    max_hypothesis_tokens=max_hypothesis_tokens,
                    max_length=max_length,
                    margin=clipped_margin,
                    contradiction_limit=clipped_contradiction_limit,
                )
            except Exception as exc:  # pragma: no cover - runtime dependency path
                runtime_heuristic_fallback = True
                LOGGER.warning(
                    "Validator NLI scoring failed for claim; switching to heuristic scoring (%s)",
                    exc,
                )
                score_payload = _heuristic_score_claim(
                    claim=claim,
                    premise=premise,
                    threshold=clipped_threshold,
                )
        else:
            score_payload = _heuristic_score_claim(
                claim=claim,
                premise=premise,
                threshold=clipped_threshold,
            )

        score_payload["premise_tokens"] = int(score_payload.get("premise_tokens", premise_info.get("tokens_used", 0)))
        score_payload["selected_chunk_count"] = int(premise_info.get("selected_chunk_count", 0))
        score_payload["selected_sentence_count"] = int(premise_info.get("selected_sentence_count", 0))
        score_payload["truncated"] = bool(score_payload.get("truncated") or premise_info.get("truncated"))
        claim_scores.append(score_payload)

        total_premise_tokens += int(score_payload.get("premise_tokens", 0))
        total_hypothesis_tokens += int(score_payload.get("hypothesis_tokens", 0))
        truncation_flag = truncation_flag or bool(score_payload.get("truncated"))

    if runtime_heuristic_fallback:
        LOGGER.warning(
            "Validator model '%s' fell back to heuristic claim scoring due to runtime inference failure.",
            effective_model,
        )
    elif not entailment_ready:
        LOGGER.warning(
            "Validator model '%s' is not entailment-tuned; running heuristic claim scoring.",
            effective_model,
        )

    aggregate = aggregate_claim_scores(
        claim_scores,
        margin=clipped_margin if entailment_ready and not runtime_heuristic_fallback else clipped_threshold * 0.5,
        contradiction_limit=clipped_contradiction_limit,
    )

    issues: list[str] = []
    if missing_pmids:
        issues.append("Answer cites PMIDs not present in retrieved sources.")
    if aggregate["details"].get("invalid_claim_count", 0):
        issues.append("One or more claims are weakly supported by retrieved evidence.")
    if truncation_flag:
        issues.append("Validator truncated evidence to fit token budget.")

    final_valid = bool(aggregate["valid"]) and not missing_pmids
    final_score = float(aggregate["score"])
    if missing_pmids:
        final_score = max(0.0, final_score - 0.25)

    details = dict(aggregate.get("details", {}))
    details.update(
        {
            "model_name": effective_model,
            "mode": (
                "nli_claim_level"
                if entailment_ready and not runtime_heuristic_fallback
                else "heuristic_claim_level"
            ),
            "runtime_heuristic_fallback": runtime_heuristic_fallback,
            "threshold": clipped_threshold,
            "margin": clipped_margin,
            "max_premise_tokens": max_premise_tokens,
            "max_hypothesis_tokens": max_hypothesis_tokens,
            "max_length": max_length,
            "top_n_chunks": top_n_chunks,
            "top_k_sentences": top_k_sentences,
            "missing_pmids": missing_pmids,
            "issues": issues,
            "premise_tokens_total": total_premise_tokens,
            "hypothesis_tokens_total": total_hypothesis_tokens,
            "truncation": truncation_flag,
        }
    )

    if claim_scores:
        details["claim_scores"] = [
            {
                "claim": item.get("claim", ""),
                "valid": bool(item.get("valid")),
                "score": float(item.get("score", 0.0)),
                "entailment": float(item.get("entailment", 0.0)),
                "neutral": float(item.get("neutral", 0.0)),
                "contradiction": float(item.get("contradiction", 0.0)),
                "truncated": bool(item.get("truncated")),
                "selected_chunk_count": int(item.get("selected_chunk_count", 0)),
                "selected_sentence_count": int(item.get("selected_sentence_count", 0)),
            }
            for item in claim_scores
        ]

    avg_score = float(aggregate["details"].get("avg_score", 0.0))
    min_score = float(aggregate["details"].get("min_score", 0.0))
    LOGGER.info(
        "[VALIDATOR] model=%s claim_count=%s avg_score=%.4f min_score=%.4f truncated=%s premise_tokens=%s hypothesis_tokens=%s",
        effective_model,
        len(claim_scores),
        avg_score,
        min_score,
        truncation_flag,
        total_premise_tokens,
        total_hypothesis_tokens,
    )
    LOGGER.info(
        "[VALIDATOR] output label=%s score=%.4f decision_valid=%s",
        aggregate["label"],
        final_score,
        final_valid,
    )

    return {
        "valid": final_valid,
        "score": final_score,
        "label": str(aggregate["label"]),
        "details": details,
    }


def _extract_evidence_chunks(retrieved_docs: list[Any] | None, context: str) -> list[str]:
    chunks: list[str] = []

    for doc in retrieved_docs or []:
        metadata = getattr(doc, "metadata", {}) or {}
        title = str(metadata.get("title", "") or "").strip()
        pmid = str(metadata.get("pmid", "") or "").strip()
        journal = str(metadata.get("journal", "") or "").strip()
        year = str(metadata.get("year", "") or "").strip()
        abstract = str(getattr(doc, "page_content", "") or "").strip()
        if not abstract:
            continue
        header_parts = []
        if pmid:
            header_parts.append(f"PMID: {pmid}")
        if title:
            header_parts.append(f"Title: {title}")
        if journal:
            header_parts.append(f"Journal: {journal}")
        if year:
            header_parts.append(f"Year: {year}")
        header = " | ".join(header_parts)
        chunks.append(f"{header}\n{abstract}" if header else abstract)

    if chunks:
        return chunks

    if not context.strip():
        return []

    sections = [segment.strip() for segment in context.split("\n\n---\n\n") if segment.strip()]
    return sections


def _heuristic_score_claim(claim: str, premise: str, threshold: float) -> dict[str, Any]:
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
    premise_tokens = set(re.findall(r"[a-z0-9]+", premise.lower()))
    overlap = len(claim_tokens & premise_tokens)
    denom = len(claim_tokens) or 1
    support = overlap / denom
    score = max(0.0, min(1.0, support))
    valid = score >= max(0.35, threshold * 0.6)
    contradiction = 1.0 - score
    return {
        "claim": claim,
        "valid": valid,
        "critical": False,
        "entailment": score,
        "neutral": 1.0 - score,
        "contradiction": contradiction,
        "raw_margin": score - contradiction,
        "score": score,
        "premise_tokens": len(premise.split()),
        "hypothesis_tokens": len(claim.split()),
        "input_tokens": len((premise + " " + claim).split()),
        "truncated": False,
        "premise": premise,
        "hypothesis": claim,
    }


def _extract_pmids(text: str) -> set[str]:
    return set(re.findall(r"\b\d{7,9}\b", text or ""))


def _clip(value: Any, *, lower: float, upper: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))
