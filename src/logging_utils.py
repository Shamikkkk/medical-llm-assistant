from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    name: str,
    *,
    logger_name: str = "app.events",
    store_path: str | Path | None = None,
    **fields: Any,
) -> dict[str, Any]:
    payload = {
        "event": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_normalize_json_fields(fields),
    }
    get_logger(logger_name).info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if store_path:
        path = Path(store_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return payload


def hash_query_text(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_usage_stats(response: Any) -> dict[str, Any]:
    usage = _extract_usage_mapping(response)
    model_name = _extract_model_name(response)
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
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model_name": model_name,
    }


def log_llm_usage(tag: str, response: Any) -> dict[str, Any]:
    logger = get_logger("llm.usage")
    usage_stats = extract_usage_stats(response)
    cost_fields = _extract_cost_fields(response)

    if (
        usage_stats["prompt_tokens"] is None
        and usage_stats["completion_tokens"] is None
        and usage_stats["total_tokens"] is None
    ):
        logger.info("[TOKENS] %s usage metadata not available", tag)
        return usage_stats

    line = (
        f"[TOKENS] {tag} "
        f"prompt={_fmt_token(usage_stats['prompt_tokens'])} "
        f"completion={_fmt_token(usage_stats['completion_tokens'])} "
        f"total={_fmt_token(usage_stats['total_tokens'])} "
        f"model={usage_stats.get('model_name') or 'unknown'}"
    )
    if cost_fields:
        line = f"{line} {cost_fields}"
    logger.info(line)
    return usage_stats


def _normalize_json_fields(fields: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        elif isinstance(value, Mapping):
            normalized[key] = _normalize_json_fields(dict(value))
        elif isinstance(value, (list, tuple)):
            normalized[key] = [
                _normalize_json_fields(item) if isinstance(item, Mapping) else _coerce_json_scalar(item)
                for item in value
            ]
        else:
            normalized[key] = _coerce_json_scalar(value)
    return normalized


def _coerce_json_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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


def _get_int(payload: Mapping[str, Any] | None, keys: list[str]) -> int | None:
    if payload is None:
        return None
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
