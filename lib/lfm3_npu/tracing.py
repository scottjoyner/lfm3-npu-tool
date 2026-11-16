"""
OpenTelemetry tracing for LFM3-1.7B inference servers.

Provides distributed tracing for prefill/decode phases across CPU and NPU
servers. Exports to OTLP-compatible collectors (e.g., Jaeger, Tempo,
OpenTelemetry Collector).
"""

import os
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.trace import SpanAttributes

# Global tracer
_tracer: Optional[trace.Tracer] = None
_initialized = False


def init_tracer(service_name: str = "lfm3-inference",
                otlp_endpoint: Optional[str] = None,
                enabled: bool = True) -> trace.Tracer:
    """Initialize the OpenTelemetry tracer.

    Args:
        service_name: Service name for telemetry.
        otlp_endpoint: OTLP collector endpoint (e.g., http://localhost:4318).
        enabled: Whether tracing is enabled. If disabled, returns a no-op tracer.
    """
    global _tracer, _initialized

    if _initialized:
        return _tracer

    _initialized = True

    if not enabled or otlp_endpoint is None:
        _tracer = trace.get_tracer(service_name)
        return _tracer

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": os.environ.get("LFM3_ENV", "production"),
    })

    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)

    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer. Initializes a no-op tracer if not yet initialized."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("lfm3-inference")
    return _tracer


@contextmanager
def trace_prefill(tracer: trace.Tracer, provider: str, model: str,
                  prompt_tokens: int):
    """Trace a prefill operation."""
    with tracer.start_as_current_span(
        "llm.prefill",
        attributes={
            SpanAttributes.LLM_REQUEST_TYPE: "prefill",
            "llm.provider": provider,
            "llm.model": model,
            "llm.prompt_tokens": prompt_tokens,
        },
    ) as span:
        start = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise
        finally:
            duration = time.perf_counter() - start
            span.set_attribute("llm.duration_seconds", duration)


@contextmanager
def trace_decode(tracer: trace.Tracer, provider: str, model: str,
                 step: int, total_tokens: int):
    """Trace a decode step."""
    with tracer.start_as_current_span(
        "llm.decode",
        attributes={
            SpanAttributes.LLM_REQUEST_TYPE: "decode",
            "llm.provider": provider,
            "llm.model": model,
            "llm.decode_step": step,
            "llm.total_tokens": total_tokens,
        },
    ) as span:
        start = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise
        finally:
            duration = time.perf_counter() - start
            span.set_attribute("llm.duration_seconds", duration)


@contextmanager
def trace_request(tracer: trace.Tracer, provider: str, model: str,
                  endpoint: str, prompt_tokens: int, max_tokens: int):
    """Trace a full inference request."""
    with tracer.start_as_current_span(
        "llm.request",
        attributes={
            SpanAttributes.LLM_REQUEST_TYPE: "request",
            "llm.provider": provider,
            "llm.model": model,
            "llm.endpoint": endpoint,
            "llm.prompt_tokens": prompt_tokens,
            "llm.max_tokens": max_tokens,
        },
    ) as span:
        start = time.perf_counter()
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.StatusCode.ERROR, str(e))
            raise
        finally:
            duration = time.perf_counter() - start
            span.set_attribute("llm.duration_seconds", duration)
