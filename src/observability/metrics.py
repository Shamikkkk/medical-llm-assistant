from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any


def read_metric_events(path: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(path).expanduser()
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def summarize_metric_events(events: list[dict[str, Any]]) -> dict[str, float | int]:
    complete_events = [event for event in events if str(event.get("event", "")) == "request.complete"]
    error_events = [event for event in events if str(event.get("event", "")) == "request.error"]
    latencies = _collect_numeric(complete_events, "total_ms")
    cache_hits = [
        bool(event.get("cache_hit"))
        for event in complete_events
        if event.get("cache_hit") is not None
    ]
    pmid_counts = _collect_numeric(complete_events, "pmid_count")
    total_requests = len(complete_events) + len(error_events)
    error_count = len(error_events)
    return {
        "total_requests": total_requests,
        "error_count": error_count,
        "error_rate": (error_count / total_requests) if total_requests else 0.0,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "cache_hit_rate": (
            sum(1 for item in cache_hits if item) / len(cache_hits)
            if cache_hits
            else 0.0
        ),
        "avg_pmid_count": (sum(pmid_counts) / len(pmid_counts)) if pmid_counts else 0.0,
    }


def _collect_numeric(events: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for event in events:
        value = event.get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (max(0, min(100, percentile)) / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float((ordered[lower] * (1 - weight)) + (ordered[upper] * weight))
