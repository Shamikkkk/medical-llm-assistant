from __future__ import annotations

import logging
import time
from typing import Any

LOGGER = logging.getLogger("validator.loader")

DEFAULT_BASE_MODEL = "microsoft/deberta-v3-base"
DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli"
PREFERRED_MNLI_MODELS: tuple[str, ...] = (
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "MoritzLaurer/DeBERTa-v3-base-mnli",
)

_MODEL_CACHE: dict[str, dict[str, Any]] = {}
_MODEL_ERRORS: dict[str, Exception] = {}
_BASE_WARNING_EMITTED = False


def get_nli_components(model_name: str | None = None) -> dict[str, Any] | None:
    requested_model = (model_name or DEFAULT_NLI_MODEL).strip() or DEFAULT_NLI_MODEL
    resolved_model = _resolve_model_name(requested_model)

    if resolved_model in _MODEL_CACHE:
        LOGGER.info("Validator model reused from cache. model=%s", resolved_model)
        payload = dict(_MODEL_CACHE[resolved_model])
        payload["from_cache"] = True
        return payload
    if resolved_model in _MODEL_ERRORS:
        return None

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency path
        _MODEL_ERRORS[resolved_model] = exc
        LOGGER.warning("Validator unavailable: import failed (%s)", exc)
        return None

    start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(resolved_model)
        model.eval()
        model.to("cpu")
    except Exception as exc:  # pragma: no cover - runtime dependency path
        _MODEL_ERRORS[resolved_model] = exc
        LOGGER.warning(
            "Validator unavailable: failed to load model=%s error=%s",
            resolved_model,
            exc,
        )
        return None

    id2label = getattr(model.config, "id2label", {}) or {}
    label_map = {int(idx): str(label) for idx, label in id2label.items()}
    entailment_ready = _is_entailment_label_map(label_map)

    payload = {
        "tokenizer": tokenizer,
        "model": model,
        "model_name": resolved_model,
        "requested_model_name": requested_model,
        "label_map": label_map,
        "entailment_ready": entailment_ready,
    }
    _MODEL_CACHE[resolved_model] = payload
    elapsed = time.perf_counter() - start
    LOGGER.info(
        "Validator model loaded (cached). requested_model=%s effective_model=%s load_seconds=%.2f",
        requested_model,
        resolved_model,
        elapsed,
    )
    return dict(payload)


def _resolve_model_name(requested_model: str) -> str:
    global _BASE_WARNING_EMITTED
    normalized = requested_model.lower().strip()
    if normalized not in {DEFAULT_BASE_MODEL, DEFAULT_NLI_MODEL.lower()}:
        return requested_model

    try:
        from transformers import AutoConfig
    except Exception:
        if not _BASE_WARNING_EMITTED:
            LOGGER.warning(
                "Validator uses base model '%s'. This is not entailment-tuned; running as heuristic validation.",
                DEFAULT_BASE_MODEL,
            )
            _BASE_WARNING_EMITTED = True
        return requested_model

    if normalized == DEFAULT_NLI_MODEL.lower():
        return DEFAULT_NLI_MODEL

    for candidate in PREFERRED_MNLI_MODELS:
        # Prefer a locally cached MNLI model first, then allow remote fetch.
        try:
            AutoConfig.from_pretrained(candidate, local_files_only=True)
            LOGGER.info(
                "Validator auto-switched to local MNLI model=%s (requested=%s)",
                candidate,
                requested_model,
            )
            return candidate
        except Exception:
            continue

    for candidate in PREFERRED_MNLI_MODELS:
        try:
            AutoConfig.from_pretrained(candidate)
            LOGGER.info(
                "Validator auto-switched to MNLI model=%s (requested=%s)",
                candidate,
                requested_model,
            )
            return candidate
        except Exception:
            continue

    if not _BASE_WARNING_EMITTED:
        LOGGER.warning(
            "Validator uses base model '%s'. This is not entailment-tuned; running as heuristic validation.",
            DEFAULT_BASE_MODEL,
        )
        _BASE_WARNING_EMITTED = True
    return requested_model


def _is_entailment_label_map(label_map: dict[int, str]) -> bool:
    lowered = {str(value).lower() for value in label_map.values()}
    return any("entail" in item for item in lowered) and any(
        "contradict" in item for item in lowered
    )
