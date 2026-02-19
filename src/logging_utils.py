from __future__ import annotations

import logging
from typing import Any, Mapping


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def log_llm_usage(tag: str, response: Any) -> None:
    logger = get_logger("llm.usage")
    usage = _extract_usage_mapping(response)
    model_name = _extract_model_name(response) or "unknown"
    cost_fields = _extract_cost_fields(response)

    if not usage:
        logger.info("[TOKENS] %s usage metadata not available", tag)
        return

    prompt_tokens = _get_int(
        usage,
        [
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
            "input_token_count",
        ],
    )
    completion_tokens = _get_int(
        usage,
        [
            "completion_tokens",
            "output_tokens",
            "completion_token_count",
            "output_token_count",
        ],
    )
    total_tokens = _get_int(usage, ["total_tokens", "total_token_count"])
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        logger.info("[TOKENS] %s usage metadata not available", tag)
        return

    line = (
        f"[TOKENS] {tag} "
        f"prompt={_fmt_token(prompt_tokens)} "
        f"completion={_fmt_token(completion_tokens)} "
        f"total={_fmt_token(total_tokens)} "
        f"model={model_name}"
    )
    if cost_fields:
        line = f"{line} {cost_fields}"
    logger.info(line)


def _extract_usage_mapping(response: Any) -> Mapping[str, Any] | None:
    for payload in _iter_metadata_payloads(response):
        if not isinstance(payload, Mapping):
            continue

        token_usage = payload.get("token_usage")
        if isinstance(token_usage, Mapping):
            return token_usage

        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            return usage

        usage_metadata = payload.get("usage_metadata")
        if isinstance(usage_metadata, Mapping):
            return usage_metadata

        if _contains_token_keys(payload):
            return payload
    return None


def _extract_model_name(response: Any) -> str | None:
    for payload in _iter_metadata_payloads(response):
        if not isinstance(payload, Mapping):
            continue
        for key in ("model_name", "model", "model_id", "model_slug"):
            value = payload.get(key)
            if value:
                return str(value)

    model_attr = getattr(response, "model", None)
    if model_attr:
        return str(model_attr)
    return None


def _extract_cost_fields(response: Any) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for payload in _iter_metadata_payloads(response):
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            key_str = str(key)
            if "cost" not in key_str.lower():
                continue
            if key_str in seen:
                continue
            seen.add(key_str)
            parts.append(f"{key_str}={value}")
    return " ".join(parts)


def _iter_metadata_payloads(response: Any):
    if response is None:
        return

    stack = [response]
    seen_ids: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)

        if isinstance(current, Mapping):
            yield current
            for key in ("response_metadata", "usage_metadata", "additional_kwargs", "llm_output"):
                nested = current.get(key)
                if nested is not None:
                    stack.append(nested)
            for key in ("token_usage", "usage"):
                nested = current.get(key)
                if isinstance(nested, Mapping):
                    stack.append(nested)
            continue

        for attr in ("response_metadata", "usage_metadata", "additional_kwargs", "llm_output"):
            nested = getattr(current, attr, None)
            if nested is not None:
                stack.append(nested)

        generations = getattr(current, "generations", None)
        if generations:
            stack.append(generations)

        message = getattr(current, "message", None)
        if message is not None:
            stack.append(message)

        if isinstance(current, (list, tuple)):
            for item in current:
                stack.append(item)


def _contains_token_keys(payload: Mapping[str, Any]) -> bool:
    keys = {
        "prompt_tokens",
        "input_tokens",
        "prompt_token_count",
        "input_token_count",
        "completion_tokens",
        "output_tokens",
        "completion_token_count",
        "output_token_count",
        "total_tokens",
        "total_token_count",
    }
    payload_keys = {str(key) for key in payload.keys()}
    return any(key in payload_keys for key in keys)


def _get_int(payload: Mapping[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _fmt_token(value: int | None) -> str:
    if value is None:
        return "na"
    return str(value)
