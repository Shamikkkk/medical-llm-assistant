# Created by Codex - Section 1

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv

from api.session_store import SessionStore
from src.core.config import AppConfig, load_config


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    load_dotenv(override=False)
    return load_config()


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    config = get_config()
    return SessionStore(config.data_dir / "sessions.json")
