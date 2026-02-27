from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import os
from typing import Any, Iterator


def tracing_enabled() -> bool:
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


@contextmanager
def start_span(name: str, *, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is None:
                continue
            try:
                span.set_attribute(key, value)
            except Exception:
                continue
        yield span


@lru_cache(maxsize=1)
def _get_tracer():
    if not tracing_enabled():
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return None

    provider = trace.get_tracer_provider()
    if provider.__class__.__name__ != "TracerProvider":
        try:
            resource = Resource.create({"service.name": "medical-llm-assistant"})
            provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip(),
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
        except Exception:
            return None
    return trace.get_tracer("medical-llm-assistant")
