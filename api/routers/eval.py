# Created by Codex - Section 1

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_config
from eval.store import EvalStore
from src.core.config import AppConfig
from src.observability.metrics import read_metric_events, summarize_metric_events

router = APIRouter(tags=["evaluation"])


@router.get("/api/eval/results")
def get_eval_results(config: AppConfig = Depends(get_config)) -> list[dict]:
    return EvalStore(config.eval_store_path).read_all()


@router.get("/api/metrics")
def get_metrics(config: AppConfig = Depends(get_config)) -> dict:
    events = read_metric_events(config.metrics_store_path)
    return summarize_metric_events(events)
