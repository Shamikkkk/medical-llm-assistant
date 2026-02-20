from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentSettings:
    enabled: bool
    use_langgraph: bool
    eval_mode: bool
    eval_sample_rate: float
    eval_store_path: Path


def load_agent_settings() -> AgentSettings:
    load_dotenv(override=False)
    enabled = _parse_bool(os.getenv("AGENT_MODE", "false"))
    use_langgraph = _parse_bool(os.getenv("AGENT_USE_LANGGRAPH", "true"))
    eval_mode = _parse_bool(os.getenv("EVAL_MODE", "false"))
    eval_sample_rate = _parse_float(os.getenv("EVAL_SAMPLE_RATE"), default=0.25)
    eval_store_path = Path(
        os.getenv("EVAL_STORE_PATH", "./data/eval/eval_results.jsonl")
    ).expanduser()
    return AgentSettings(
        enabled=enabled,
        use_langgraph=use_langgraph,
        eval_mode=eval_mode,
        eval_sample_rate=max(0.0, min(1.0, eval_sample_rate)),
        eval_store_path=eval_store_path,
    )


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default
