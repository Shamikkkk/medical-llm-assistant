from __future__ import annotations

import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA

DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"


def get_nvidia_llm() -> ChatNVIDIA:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not set.")

    model_name = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
    return ChatNVIDIA(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )
