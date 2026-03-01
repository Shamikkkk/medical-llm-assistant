# Created by Codex - Section 1

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_config
from src.core.config import AppConfig

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
def read_config(config: AppConfig = Depends(get_config)) -> dict:
    return config.masked_summary()
