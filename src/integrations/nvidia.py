from __future__ import annotations

from functools import lru_cache
import os
from threading import Lock

from langchain_nvidia_ai_endpoints import ChatNVIDIA

DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

_LLM_BUILD_LOCK = Lock()


def get_nvidia_llm(
    *,
    model_name: str | None = None,
    api_key: str | None = None,
) -> ChatNVIDIA:
    resolved_api_key = str(api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
    if not resolved_api_key:
        raise ValueError("NVIDIA_API_KEY is not set.")
    resolved_model = str(model_name or os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)).strip()
    if not resolved_model:
        resolved_model = DEFAULT_NVIDIA_MODEL
    return _get_nvidia_llm_cached(resolved_model, resolved_api_key)


@lru_cache(maxsize=8)
def _get_nvidia_llm_cached(model_name: str, api_key: str) -> ChatNVIDIA:
    with _LLM_BUILD_LOCK:
        return ChatNVIDIA(
            model=model_name,
            temperature=0,
            api_key=api_key,
            streaming=True,
        )
