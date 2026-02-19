from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    app_title: str
    app_description: str
    nvidia_api_key: str | None
    data_dir: Path
    log_level: str
    use_reranker: bool
    log_pipeline: bool
    validator_enabled: bool
    validator_model_name: str
    validator_threshold: float
    validator_margin: float
    validator_max_premise_tokens: int
    validator_max_hypothesis_tokens: int
    validator_max_length: int
    validator_top_n_chunks: int
    validator_top_k_sentences: int


def load_config() -> AppConfig:
    """Load environment variables and return app configuration."""
    load_dotenv(override=False)

    app_title = os.getenv("APP_TITLE", "Cardio PubMed Assistant")
    app_description = os.getenv(
        "APP_DESCRIPTION",
        "Cardiovascular-focused PubMed conversational assistant.",
    )
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().absolute()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    use_reranker = _parse_bool(os.getenv("USE_RERANKER", "true"))
    log_pipeline = _parse_bool(os.getenv("LOG_PIPELINE", "false"))
    validator_enabled = _parse_bool(os.getenv("VALIDATOR_ENABLED", "false"))
    validator_model_name = (
        os.getenv("VALIDATOR_MODEL_NAME", "MoritzLaurer/DeBERTa-v3-base-mnli").strip()
        or "MoritzLaurer/DeBERTa-v3-base-mnli"
    )
    validator_threshold = _parse_float(os.getenv("VALIDATOR_THRESHOLD"), default=0.7)
    validator_margin = _parse_float(os.getenv("VALIDATOR_MARGIN"), default=0.2)
    validator_max_premise_tokens = _parse_int(os.getenv("VALIDATOR_MAX_PREMISE_TOKENS"), default=384)
    validator_max_hypothesis_tokens = _parse_int(
        os.getenv("VALIDATOR_MAX_HYPOTHESIS_TOKENS"), default=128
    )
    validator_max_length = _parse_int(os.getenv("VALIDATOR_MAX_LENGTH"), default=512)
    validator_top_n_chunks = _parse_int(os.getenv("VALIDATOR_TOP_N_CHUNKS"), default=4)
    validator_top_k_sentences = _parse_int(os.getenv("VALIDATOR_TOP_K_SENTENCES"), default=2)

    return AppConfig(
        app_title=app_title,
        app_description=app_description,
        nvidia_api_key=nvidia_api_key,
        data_dir=data_dir,
        log_level=log_level,
        use_reranker=use_reranker,
        log_pipeline=log_pipeline,
        validator_enabled=validator_enabled,
        validator_model_name=validator_model_name,
        validator_threshold=max(0.0, min(1.0, validator_threshold)),
        validator_margin=max(0.0, min(1.0, validator_margin)),
        validator_max_premise_tokens=max(64, min(1024, validator_max_premise_tokens)),
        validator_max_hypothesis_tokens=max(32, min(512, validator_max_hypothesis_tokens)),
        validator_max_length=max(128, min(1024, validator_max_length)),
        validator_top_n_chunks=max(1, min(10, validator_top_n_chunks)),
        validator_top_k_sentences=max(1, min(5, validator_top_k_sentences)),
    )


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default
