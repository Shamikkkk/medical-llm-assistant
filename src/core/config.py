from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

from dotenv import load_dotenv

DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"


class ConfigValidationError(ValueError):
    """Raised when enabled features are missing required configuration."""


@dataclass(frozen=True)
class AppConfig:
    app_title: str
    app_description: str
    nvidia_api_key: str | None
    nvidia_model: str
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
    agent_mode: bool
    agent_use_langgraph: bool
    eval_mode: bool
    eval_sample_rate: float
    eval_store_path: Path
    pubmed_cache_ttl_seconds: int
    pubmed_negative_cache_ttl_seconds: int
    max_abstracts: int
    max_context_abstracts: int
    max_context_tokens: int
    context_trim_strategy: str
    multi_strategy_retrieval: bool
    retrieval_candidate_multiplier: int
    hybrid_retrieval: bool
    hybrid_alpha: float
    citation_alignment: bool
    alignment_mode: str
    show_rewritten_query: bool
    auto_scroll: bool
    answer_cache_ttl_seconds: int
    answer_cache_min_similarity: float
    answer_cache_strict_fingerprint: bool
    metrics_mode: bool
    metrics_store_path: Path
    config_errors: tuple[str, ...]
    config_warnings: tuple[str, ...]

    def masked_summary(self) -> dict[str, Any]:
        return {
            "app_title": self.app_title,
            "app_description": self.app_description,
            "nvidia_api_key": "SET" if self.nvidia_api_key else "NOT_SET",
            "nvidia_model": self.nvidia_model,
            "data_dir": str(self.data_dir),
            "log_level": self.log_level,
            "use_reranker": self.use_reranker,
            "validator_enabled": self.validator_enabled,
            "agent_mode": self.agent_mode,
            "agent_use_langgraph": self.agent_use_langgraph,
            "eval_mode": self.eval_mode,
            "eval_store_path": str(self.eval_store_path),
            "pubmed_cache_ttl_seconds": self.pubmed_cache_ttl_seconds,
            "pubmed_negative_cache_ttl_seconds": self.pubmed_negative_cache_ttl_seconds,
            "max_abstracts": self.max_abstracts,
            "max_context_abstracts": self.max_context_abstracts,
            "max_context_tokens": self.max_context_tokens,
            "context_trim_strategy": self.context_trim_strategy,
            "multi_strategy_retrieval": self.multi_strategy_retrieval,
            "retrieval_candidate_multiplier": self.retrieval_candidate_multiplier,
            "hybrid_retrieval": self.hybrid_retrieval,
            "hybrid_alpha": self.hybrid_alpha,
            "citation_alignment": self.citation_alignment,
            "alignment_mode": self.alignment_mode,
            "show_rewritten_query": self.show_rewritten_query,
            "auto_scroll": self.auto_scroll,
            "answer_cache_ttl_seconds": self.answer_cache_ttl_seconds,
            "answer_cache_min_similarity": self.answer_cache_min_similarity,
            "answer_cache_strict_fingerprint": self.answer_cache_strict_fingerprint,
            "metrics_mode": self.metrics_mode,
            "metrics_store_path": str(self.metrics_store_path),
            "config_errors": list(self.config_errors),
            "config_warnings": list(self.config_warnings),
        }

    def require_valid(self) -> None:
        if self.config_errors:
            raise ConfigValidationError("\n".join(self.config_errors))


def load_config() -> AppConfig:
    """Load environment variables and return app configuration."""
    load_dotenv(override=False)

    app_title = os.getenv("APP_TITLE", "PubMed Literature Assistant")
    app_description = os.getenv(
        "APP_DESCRIPTION",
        "General medical PubMed conversational assistant.",
    )
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_model = os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL).strip() or DEFAULT_NVIDIA_MODEL
    data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().absolute()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    use_reranker = _parse_bool(os.getenv("USE_RERANKER", "false"))
    log_pipeline = _parse_bool(os.getenv("LOG_PIPELINE", "false"))
    validator_enabled = _parse_bool(os.getenv("VALIDATOR_ENABLED", "false"))
    validator_model_name = (
        os.getenv("VALIDATOR_MODEL_NAME", "MoritzLaurer/DeBERTa-v3-base-mnli").strip()
        or "MoritzLaurer/DeBERTa-v3-base-mnli"
    )
    validator_threshold = _parse_float(os.getenv("VALIDATOR_THRESHOLD"), default=0.7)
    validator_margin = _parse_float(os.getenv("VALIDATOR_MARGIN"), default=0.2)
    validator_max_premise_tokens = _parse_int(
        os.getenv("VALIDATOR_MAX_PREMISE_TOKENS"), default=384
    )
    validator_max_hypothesis_tokens = _parse_int(
        os.getenv("VALIDATOR_MAX_HYPOTHESIS_TOKENS"), default=128
    )
    validator_max_length = _parse_int(os.getenv("VALIDATOR_MAX_LENGTH"), default=512)
    validator_top_n_chunks = _parse_int(os.getenv("VALIDATOR_TOP_N_CHUNKS"), default=4)
    validator_top_k_sentences = _parse_int(
        os.getenv("VALIDATOR_TOP_K_SENTENCES"), default=2
    )
    agent_mode = _parse_bool(os.getenv("AGENT_MODE", "false"))
    agent_use_langgraph = _parse_bool(os.getenv("AGENT_USE_LANGGRAPH", "true"))
    eval_mode = _parse_bool(os.getenv("EVAL_MODE", "false"))
    eval_sample_rate = _parse_float(os.getenv("EVAL_SAMPLE_RATE"), default=0.25)
    eval_store_path = Path(
        os.getenv("EVAL_STORE_PATH", "./data/eval/eval_results.jsonl")
    ).expanduser().absolute()
    pubmed_cache_ttl_seconds = _parse_int(
        os.getenv("PUBMED_CACHE_TTL_SECONDS"), default=604800
    )
    pubmed_negative_cache_ttl_seconds = _parse_int(
        os.getenv("PUBMED_NEGATIVE_CACHE_TTL_SECONDS"), default=3600
    )
    legacy_max_abstracts_raw = os.getenv("MAX_ABSTRACTS")
    max_context_abstracts_raw = os.getenv("MAX_CONTEXT_ABSTRACTS")
    if max_context_abstracts_raw is not None:
        max_context_abstracts = _parse_int(max_context_abstracts_raw, default=8)
        if legacy_max_abstracts_raw is not None:
            config_note = (
                "MAX_ABSTRACTS is deprecated and is treated as a legacy alias for MAX_CONTEXT_ABSTRACTS. "
                "Using MAX_CONTEXT_ABSTRACTS."
            )
            config_warnings = [config_note]
        else:
            config_warnings = []
    else:
        max_context_abstracts = _parse_int(legacy_max_abstracts_raw, default=8)
        config_warnings = []
        if legacy_max_abstracts_raw is not None:
            config_warnings.append(
                "MAX_ABSTRACTS is deprecated. Treating it as MAX_CONTEXT_ABSTRACTS (prompt context budget)."
            )
    max_abstracts = max_context_abstracts
    max_context_tokens = _parse_int(os.getenv("MAX_CONTEXT_TOKENS"), default=2500)
    context_trim_strategy = (
        os.getenv("CONTEXT_TRIM_STRATEGY", "truncate").strip().lower() or "truncate"
    )
    multi_strategy_retrieval = _parse_bool(os.getenv("MULTI_STRATEGY_RETRIEVAL", "true"))
    retrieval_candidate_multiplier = _parse_int(
        os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER"),
        default=3,
    )
    hybrid_retrieval = _parse_bool(os.getenv("HYBRID_RETRIEVAL", "false"))
    hybrid_alpha = _parse_float(os.getenv("HYBRID_ALPHA"), default=0.5)
    citation_alignment = _parse_bool(os.getenv("CITATION_ALIGNMENT", "true"))
    alignment_mode = os.getenv("ALIGNMENT_MODE", "disclaim").strip().lower() or "disclaim"
    show_rewritten_query = _parse_bool(os.getenv("SHOW_REWRITTEN_QUERY", "false"))
    auto_scroll = _parse_bool(os.getenv("AUTO_SCROLL", "true"))
    answer_cache_ttl_seconds = _parse_int(
        os.getenv("ANSWER_CACHE_TTL_SECONDS"),
        default=604800,
    )
    answer_cache_min_similarity = _parse_float(
        os.getenv("ANSWER_CACHE_MIN_SIMILARITY"),
        default=0.9,
    )
    answer_cache_strict_fingerprint = _parse_bool(
        os.getenv("ANSWER_CACHE_STRICT_FINGERPRINT", "true")
    )
    metrics_mode = _parse_bool(os.getenv("METRICS_MODE", "false"))
    metrics_store_path = Path(
        os.getenv("METRICS_STORE_PATH", "./data/metrics/events.jsonl")
    ).expanduser().absolute()

    config_errors: list[str] = []
    config_warnings = list(config_warnings)
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not nvidia_api_key:
        config_warnings.append(
            "NVIDIA_API_KEY is not set. PubMed retrieval will work, but LLM answer generation is disabled."
        )
    if eval_mode and not openai_api_key:
        config_errors.append("EVAL_MODE=true requires OPENAI_API_KEY.")
    if eval_mode and not openai_base_url:
        config_errors.append("EVAL_MODE=true requires OPENAI_BASE_URL.")
    if context_trim_strategy not in {"truncate", "compress"}:
        config_warnings.append(
            f"Invalid CONTEXT_TRIM_STRATEGY='{context_trim_strategy}'. Falling back to 'truncate'."
        )
        context_trim_strategy = "truncate"
    if alignment_mode not in {"disclaim", "remove"}:
        config_warnings.append(
            f"Invalid ALIGNMENT_MODE='{alignment_mode}'. Falling back to 'disclaim'."
        )
        alignment_mode = "disclaim"

    return AppConfig(
        app_title=app_title,
        app_description=app_description,
        nvidia_api_key=nvidia_api_key,
        nvidia_model=nvidia_model,
        data_dir=data_dir,
        log_level=log_level,
        use_reranker=use_reranker,
        log_pipeline=log_pipeline,
        validator_enabled=validator_enabled,
        validator_model_name=validator_model_name,
        validator_threshold=max(0.0, min(1.0, validator_threshold)),
        validator_margin=max(0.0, min(1.0, validator_margin)),
        validator_max_premise_tokens=max(64, min(1024, validator_max_premise_tokens)),
        validator_max_hypothesis_tokens=max(
            32, min(512, validator_max_hypothesis_tokens)
        ),
        validator_max_length=max(128, min(1024, validator_max_length)),
        validator_top_n_chunks=max(1, min(10, validator_top_n_chunks)),
        validator_top_k_sentences=max(1, min(5, validator_top_k_sentences)),
        agent_mode=agent_mode,
        agent_use_langgraph=agent_use_langgraph,
        eval_mode=eval_mode,
        eval_sample_rate=max(0.0, min(1.0, eval_sample_rate)),
        eval_store_path=eval_store_path,
        pubmed_cache_ttl_seconds=max(60, pubmed_cache_ttl_seconds),
        pubmed_negative_cache_ttl_seconds=max(60, pubmed_negative_cache_ttl_seconds),
        max_abstracts=max(1, min(20, max_abstracts)),
        max_context_abstracts=max(1, min(20, max_context_abstracts)),
        max_context_tokens=max(256, min(20000, max_context_tokens)),
        context_trim_strategy=context_trim_strategy,
        multi_strategy_retrieval=multi_strategy_retrieval,
        retrieval_candidate_multiplier=max(1, min(10, retrieval_candidate_multiplier)),
        hybrid_retrieval=hybrid_retrieval,
        hybrid_alpha=max(0.0, min(1.0, hybrid_alpha)),
        citation_alignment=citation_alignment,
        alignment_mode=alignment_mode,
        show_rewritten_query=show_rewritten_query,
        auto_scroll=auto_scroll,
        answer_cache_ttl_seconds=max(60, answer_cache_ttl_seconds),
        answer_cache_min_similarity=max(0.0, min(1.0, answer_cache_min_similarity)),
        answer_cache_strict_fingerprint=answer_cache_strict_fingerprint,
        metrics_mode=metrics_mode,
        metrics_store_path=metrics_store_path,
        config_errors=tuple(config_errors),
        config_warnings=tuple(config_warnings),
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
