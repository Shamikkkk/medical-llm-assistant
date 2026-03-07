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


@router.get("/api/monitor/overview")
def get_monitor_overview(config: AppConfig = Depends(get_config)) -> dict:
    """Combined overview for the monitoring dashboard."""
    events = read_metric_events(config.metrics_store_path)
    metrics = summarize_metric_events(events)

    eval_records = EvalStore(config.eval_store_path).read_all()
    recent_evals = eval_records[-50:] if eval_records else []

    # Aggregate eval scores across recent records
    score_fields = [
        "faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall",
        "citation_alignment",
        "safety_compliance",
    ]
    agg_scores: dict[str, float] = {}
    for field in score_fields:
        values = [
            float(rec[field])
            for rec in recent_evals
            if isinstance(rec.get(field), (int, float))
        ]
        agg_scores[field] = round(sum(values) / len(values), 4) if values else 0.0

    # Pipeline step latencies from recent complete events
    complete_events = [e for e in events if e.get("event") == "request.complete"]
    recent_complete = complete_events[-100:]
    step_latencies: dict[str, list[float]] = {}
    for ev in recent_complete:
        for key, val in (ev.get("timings") or {}).items():
            try:
                step_latencies.setdefault(key, []).append(float(val))
            except (TypeError, ValueError):
                pass

    step_avg: dict[str, float] = {
        k: round(sum(v) / len(v), 1) for k, v in step_latencies.items() if v
    }

    # Agent mode usage
    agent_count = sum(1 for ev in complete_events if ev.get("agent_mode"))
    pipeline_count = len(complete_events) - agent_count

    return {
        "metrics": metrics,
        "avg_eval_scores": agg_scores,
        "step_avg_latencies_ms": step_avg,
        "agent_mode_count": agent_count,
        "pipeline_mode_count": pipeline_count,
        "total_eval_records": len(eval_records),
    }


@router.get("/api/monitor/recent-evals")
def get_recent_evals(
    limit: int = 20,
    config: AppConfig = Depends(get_config),
) -> list[dict]:
    """Return the N most recent evaluation records."""
    records = EvalStore(config.eval_store_path).read_all()
    return records[-max(1, min(200, limit)):]


@router.get("/api/monitor/latency-series")
def get_latency_series(
    limit: int = 50,
    config: AppConfig = Depends(get_config),
) -> list[dict]:
    """Return recent request latency data points for charting."""
    events = read_metric_events(config.metrics_store_path)
    complete = [e for e in events if e.get("event") == "request.complete"]
    recent = complete[-max(1, min(500, limit)):]
    return [
        {
            "ts": ev.get("ts") or ev.get("timestamp", ""),
            "total_ms": ev.get("total_ms", 0),
            "cache_hit": bool(ev.get("cache_hit")),
            "agent_mode": bool(ev.get("agent_mode")),
        }
        for ev in recent
    ]
