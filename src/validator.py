from __future__ import annotations

import logging
import time
from typing import Any

LOGGER = logging.getLogger("validator.loader")
DEFAULT_VALIDATOR_MODEL = "microsoft/deberta-v3-base"
_VALIDATOR_PIPES: dict[str, Any] = {}
_VALIDATOR_LOAD_ERRORS: dict[str, Exception] = {}
_BASE_MODEL_WARNING_EMITTED = False


def get_validator(model_name: str | None = None):
    requested_model = (model_name or DEFAULT_VALIDATOR_MODEL).strip() or DEFAULT_VALIDATOR_MODEL
    effective_model = _resolve_model_name(requested_model)

    if effective_model in _VALIDATOR_PIPES:
        LOGGER.info("Validator model reused from cache. model=%s", effective_model)
        return {
            "pipeline": _VALIDATOR_PIPES[effective_model],
            "model_name": effective_model,
            "requested_model_name": requested_model,
            "from_cache": True,
        }
    if effective_model in _VALIDATOR_LOAD_ERRORS:
        return None

    start = time.perf_counter()
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover - optional dependency path
        _VALIDATOR_LOAD_ERRORS[effective_model] = exc
        LOGGER.warning("Validator unavailable: transformers import failed (%s)", exc)
        return None

    try:
        pipe = pipeline(
            "text-classification",
            model=effective_model,
            device=-1,
        )
        _VALIDATOR_PIPES[effective_model] = pipe
        elapsed = time.perf_counter() - start
        LOGGER.info(
            "Validator model loaded (cached). requested_model=%s effective_model=%s load_seconds=%.2f",
            requested_model,
            effective_model,
            elapsed,
        )
        return {
            "pipeline": pipe,
            "model_name": effective_model,
            "requested_model_name": requested_model,
            "from_cache": False,
        }
    except Exception as exc:  # pragma: no cover - model load depends on runtime
        _VALIDATOR_LOAD_ERRORS[effective_model] = exc
        LOGGER.warning("Validator unavailable: model load failed model=%s error=%s", effective_model, exc)
        return None


def _resolve_model_name(requested_model: str) -> str:
    """Prefer locally-available MNLI variants when requested model is base DeBERTa."""
    global _BASE_MODEL_WARNING_EMITTED
    requested = requested_model.strip() or DEFAULT_VALIDATOR_MODEL
    normalized = requested.lower()
    if normalized != DEFAULT_VALIDATOR_MODEL:
        return requested

    mnli_candidates = (
        "microsoft/deberta-v3-base-mnli",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    )
    try:
        from transformers import AutoConfig
    except Exception:
        if not _BASE_MODEL_WARNING_EMITTED:
            LOGGER.warning(
                "Validator uses base model '%s'. This is not entailment-tuned; running as heuristic validation.",
                DEFAULT_VALIDATOR_MODEL,
            )
            _BASE_MODEL_WARNING_EMITTED = True
        return requested

    for candidate in mnli_candidates:
        try:
            AutoConfig.from_pretrained(candidate, local_files_only=True)
            LOGGER.info(
                "Validator auto-switched to locally available MNLI model=%s (requested=%s)",
                candidate,
                requested,
            )
            return candidate
        except Exception:
            continue

    if not _BASE_MODEL_WARNING_EMITTED:
        LOGGER.warning(
            "Validator uses base model '%s'. This is not entailment-tuned; running as heuristic validation.",
            DEFAULT_VALIDATOR_MODEL,
        )
        _BASE_MODEL_WARNING_EMITTED = True
    return requested
