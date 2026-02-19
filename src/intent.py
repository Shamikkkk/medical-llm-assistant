from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import json
import logging
import re
from typing import Any, Literal

LOGGER = logging.getLogger("pipeline.intent")

IntentLabel = Literal["smalltalk", "medical"]

# Broad biomedical terms (not greeting lists) to avoid misrouting real questions.
MEDICAL_KEYWORDS: tuple[str, ...] = (
    "cardio",
    "cardiac",
    "heart",
    "hypertension",
    "atrial",
    "coronary",
    "myocard",
    "stroke",
    "heart failure",
    "arrhythmia",
    "ecg",
    "ekg",
    "copd",
    "pulmonary",
    "disease",
    "treatment",
    "therapy",
    "diagnosis",
    "symptom",
    "drug",
    "medication",
    "dose",
    "trial",
    "mortality",
    "risk",
    "patient",
)

_LLM_CACHE_MAX = 512
_LLM_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def classify_intent(user_text: str, llm: Any | None) -> IntentLabel:
    details = classify_intent_details(user_text, llm=llm, log_enabled=False)
    label = str(details.get("label", "medical")).lower()
    return "smalltalk" if label == "smalltalk" else "medical"


def classify_intent_details(
    user_text: str, llm: Any | None, *, log_enabled: bool = False
) -> dict[str, Any]:
    normalized = _normalize(user_text)
    if not normalized:
        result = {"label": "smalltalk", "confidence": 0.9, "reason": "empty_input"}
        _log_result(result, source="heuristic", cache_hit=False, enabled=log_enabled, query=normalized)
        return result

    cached = _llm_cache_get(normalized)
    if cached is not None:
        _log_result(cached, source="llm", cache_hit=True, enabled=log_enabled, query=normalized)
        return dict(cached)

    if llm is not None:
        llm_result = _classify_with_llm(user_text, llm)
        if llm_result is not None:
            _llm_cache_put(normalized, llm_result)
            _log_result(llm_result, source="llm", cache_hit=False, enabled=log_enabled, query=normalized)
            return llm_result

    cache_before = _heuristic_label.cache_info()
    label, confidence, reason = _heuristic_label(normalized)
    cache_after = _heuristic_label.cache_info()
    result = {"label": label, "confidence": confidence, "reason": reason}
    _log_result(
        result,
        source="heuristic",
        cache_hit=cache_after.hits > cache_before.hits,
        enabled=log_enabled,
        query=normalized,
    )
    return result


def smalltalk_reply(query: str, llm: Any | None = None) -> str:
    del llm  # deterministic smalltalk keeps this path cheap and robust
    normalized = _normalize(query)
    if "help" in normalized and "you" in normalized:
        return (
            "I can help with cardiovascular and cardiopulmonary overlap literature questions. "
            "Ask a clinical question and I will search PubMed abstracts, summarize findings, "
            "and provide PMID-linked sources."
        )
    if "thank" in normalized:
        return "You're welcome. Ask me any cardiovascular evidence question when you're ready."
    return (
        "Hi. I can help with cardiovascular and cardiopulmonary overlap PubMed questions. "
        "For example: 'DOACs vs warfarin for stroke prevention in atrial fibrillation.'"
    )


def _classify_with_llm(user_text: str, llm: Any) -> dict[str, Any] | None:
    prompt = (
        "You are a classifier. Output ONLY one token: smalltalk OR medical.\n"
        f"Message: {user_text}\n"
    )
    raw = _invoke_llm(llm, prompt)
    if not raw:
        return None
    label = _parse_llm_label(raw)
    if label is None:
        # Some models return JSON despite token instruction. Support that too.
        parsed = _parse_json_block(raw)
        if parsed is None:
            return None
        candidate = str(parsed.get("label", "")).strip().lower()
        label = "smalltalk" if candidate == "smalltalk" else "medical" if candidate == "medical" else None
    if label is None:
        return None
    return {"label": label, "confidence": 0.9, "reason": "llm_intent"}


@lru_cache(maxsize=1024)
def _heuristic_label(normalized_query: str) -> tuple[IntentLabel, float, str]:
    token_count = len(normalized_query.split())
    has_medical = _contains_medical_keyword(normalized_query)
    if has_medical:
        return ("medical", 0.78, "heuristic_medical_keyword")
    if token_count <= 3:
        return ("smalltalk", 0.8, "heuristic_short_non_medical")
    # Short capability/meta questions can still be treated as smalltalk without fixed greeting lists.
    if token_count <= 6 and ("you" in normalized_query or "help" in normalized_query):
        return ("smalltalk", 0.72, "heuristic_short_meta")
    return ("medical", 0.6, "heuristic_default_medical_route")


def _contains_medical_keyword(text: str) -> bool:
    for keyword in MEDICAL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return True
    return False


def _invoke_llm(llm: Any, prompt: str) -> str | None:
    try:
        if hasattr(llm, "invoke"):
            result = llm.invoke(prompt)
            return _extract_text(result)
        if hasattr(llm, "predict"):
            result = llm.predict(prompt)
            return _extract_text(result)
    except Exception:
        return None
    return None


def _extract_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(result or "")


def _parse_llm_label(text: str) -> IntentLabel | None:
    normalized = _normalize(text)
    if normalized == "smalltalk":
        return "smalltalk"
    if normalized == "medical":
        return "medical"
    if "smalltalk" in normalized and "medical" not in normalized:
        return "smalltalk"
    if "medical" in normalized and "smalltalk" not in normalized:
        return "medical"
    return None


def _parse_json_block(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _normalize(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[\W_]+", " ", lowered)
    return " ".join(lowered.split())


def _llm_cache_get(normalized: str) -> dict[str, Any] | None:
    payload = _LLM_CACHE.get(normalized)
    if payload is None:
        return None
    _LLM_CACHE.move_to_end(normalized)
    return payload


def _llm_cache_put(normalized: str, payload: dict[str, Any]) -> None:
    _LLM_CACHE[normalized] = dict(payload)
    _LLM_CACHE.move_to_end(normalized)
    while len(_LLM_CACHE) > _LLM_CACHE_MAX:
        _LLM_CACHE.popitem(last=False)


def _log_result(
    result: dict[str, Any],
    *,
    source: str,
    cache_hit: bool,
    enabled: bool,
    query: str,
) -> None:
    if not enabled:
        return
    LOGGER.info(
        "[PIPELINE] Intent classification | label=%s confidence=%.2f source=%s cache_hit=%s query='%s'",
        result.get("label", "unknown"),
        _coerce_confidence(result.get("confidence", 0.0)),
        source,
        cache_hit,
        _trim(query),
    )


def _coerce_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _trim(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
