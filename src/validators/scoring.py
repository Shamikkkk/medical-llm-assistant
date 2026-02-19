from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any

_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_CRITICAL_HINTS = (
    "reduce",
    "increase",
    "associated",
    "significant",
    "mortality",
    "bleeding",
    "stroke",
    "risk",
    "odds",
    "hazard",
    "trial",
    "outcome",
)


def score_claim_with_nli(
    *,
    claim: str,
    premise: str,
    tokenizer: Any,
    model: Any,
    label_map: dict[int, str],
    max_premise_tokens: int,
    max_hypothesis_tokens: int,
    max_length: int,
    margin: float,
    contradiction_limit: float,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(f"torch dependency missing for NLI scoring: {exc}") from exc

    premise_text, premise_tokens, premise_truncated = _clip_text_to_tokens(
        premise,
        tokenizer=tokenizer,
        token_budget=max(32, int(max_premise_tokens)),
    )
    hypothesis_text, hypothesis_tokens, hypothesis_truncated = _clip_text_to_tokens(
        claim,
        tokenizer=tokenizer,
        token_budget=max(16, int(max_hypothesis_tokens)),
    )

    model_inputs = tokenizer(
        premise_text,
        hypothesis_text,
        truncation=True,
        max_length=max(128, int(max_length)),
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**model_inputs).logits[0]
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()

    label_indices = _resolve_label_indices(label_map, len(probabilities))
    contradiction = float(probabilities[label_indices["contradiction"]])
    neutral = float(probabilities[label_indices["neutral"]])
    entailment = float(probabilities[label_indices["entailment"]])

    margin_raw = entailment - contradiction
    normalized_score = _clamp01((margin_raw + 1.0) / 2.0)
    critical_claim = _is_critical_claim(claim)
    effective_margin = margin if critical_claim else margin * 0.6
    effective_contradiction = contradiction_limit if critical_claim else min(
        0.35, contradiction_limit + 0.1
    )
    valid = margin_raw >= effective_margin and contradiction <= effective_contradiction

    return {
        "claim": claim,
        "valid": valid,
        "critical": critical_claim,
        "entailment": entailment,
        "neutral": neutral,
        "contradiction": contradiction,
        "raw_margin": margin_raw,
        "score": normalized_score,
        "premise_tokens": premise_tokens,
        "hypothesis_tokens": hypothesis_tokens,
        "input_tokens": int(model_inputs["input_ids"].shape[-1]),
        "truncated": bool(premise_truncated or hypothesis_truncated),
        "premise": premise_text,
        "hypothesis": hypothesis_text,
    }


def aggregate_claim_scores(
    claim_scores: list[dict[str, Any]],
    *,
    margin: float,
    contradiction_limit: float,
) -> dict[str, Any]:
    if not claim_scores:
        return {
            "valid": True,
            "score": 0.0,
            "label": "NO_CLAIMS",
            "details": {
                "reason": "No claims were available for validation.",
                "claim_count": 0,
            },
        }

    avg_score = float(mean(item["score"] for item in claim_scores))
    min_score = float(min(item["score"] for item in claim_scores))
    avg_margin = float(mean(item["raw_margin"] for item in claim_scores))
    max_contradiction = float(max(item["contradiction"] for item in claim_scores))
    truncation_flag = any(bool(item.get("truncated")) for item in claim_scores)
    critical_failures = sum(
        1 for item in claim_scores if bool(item.get("critical")) and not bool(item.get("valid"))
    )
    invalid_count = sum(1 for item in claim_scores if not bool(item.get("valid")))

    decision_valid = (
        avg_margin >= margin
        and max_contradiction <= contradiction_limit
        and critical_failures == 0
    )
    label = "HEURISTIC_OK" if decision_valid else "HEURISTIC_WARNING"

    return {
        "valid": decision_valid,
        "score": avg_score,
        "label": label,
        "details": {
            "reason": (
                "Claims are supported by retrieved evidence."
                if decision_valid
                else "Some claims are weakly supported or contradicted."
            ),
            "claim_count": len(claim_scores),
            "invalid_claim_count": invalid_count,
            "critical_failures": critical_failures,
            "avg_score": avg_score,
            "min_score": min_score,
            "avg_margin": avg_margin,
            "max_contradiction": max_contradiction,
            "truncation": truncation_flag,
        },
    }


def _resolve_label_indices(label_map: dict[int, str], n_classes: int) -> dict[str, int]:
    normalized = {idx: str(name).lower() for idx, name in label_map.items()}
    contradiction = _find_label_index(normalized, "contradict")
    neutral = _find_label_index(normalized, "neutral")
    entailment = _find_label_index(normalized, "entail")

    if contradiction is None or neutral is None or entailment is None:
        # Common MNLI order fallback.
        fallback = [0, 1, 2]
        if n_classes < 3:
            fallback = list(range(max(1, n_classes))) + [0, 0]
        contradiction = fallback[0]
        neutral = fallback[1]
        entailment = fallback[2]

    contradiction = int(max(0, min(n_classes - 1, contradiction)))
    neutral = int(max(0, min(n_classes - 1, neutral)))
    entailment = int(max(0, min(n_classes - 1, entailment)))
    return {
        "contradiction": contradiction,
        "neutral": neutral,
        "entailment": entailment,
    }


def _find_label_index(labels: dict[int, str], needle: str) -> int | None:
    for idx, label in labels.items():
        if needle in label:
            return idx
    return None


def _clip_text_to_tokens(text: str, *, tokenizer: Any, token_budget: int) -> tuple[str, int, bool]:
    token_budget = max(8, int(token_budget))
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= token_budget:
            return text, len(ids), False
        clipped_ids = ids[:token_budget]
        clipped = tokenizer.decode(
            clipped_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        return clipped, len(clipped_ids), True
    except Exception:
        words = str(text).split()
        if len(words) <= token_budget:
            return text, len(words), False
        clipped = " ".join(words[:token_budget]).strip()
        return clipped, token_budget, True


def _is_critical_claim(claim: str) -> bool:
    lowered = str(claim).lower()
    if _NUMERIC_TOKEN_RE.search(lowered):
        return True
    return any(term in lowered for term in _CRITICAL_HINTS)


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))
