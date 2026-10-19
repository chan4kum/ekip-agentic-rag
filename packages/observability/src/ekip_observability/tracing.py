"""Distributed tracing (OpenTelemetry). The provider is built here and passed in (no globals)."""

from typing import Any, Literal

from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

Exporter = Literal["none", "console"]

# Attributes that can carry the request URL. Names differ between semantic-convention
# versions (old http.* vs new url.*), so we scrub all of them.
_URL_ATTRIBUTES = ("http.url", "http.target", "url.full", "url.path")


def scrub_url_attributes(span: Any) -> None:
    """Strip ?query=... from the span's URL attributes (works on any recording span)."""
    attributes = getattr(span, "attributes", None) or {}
    for key in _URL_ATTRIBUTES:
        value = attributes.get(key)
        if isinstance(value, str) and "?" in value:
            span.set_attribute(key, value.split("?", 1)[0])
    if "url.query" in attributes:
        span.set_attribute("url.query", "[redacted]")


class ScrubbingSpanProcessor(SpanProcessor):
    """Scrub attributes that are present when a span STARTS (e.g. outbound HTTP client spans).

    The FastAPI/ASGI instrumentation adds the request URL AFTER the span starts, so for server
    spans the same function is also called from the instrumentation's request hook (see main.py).
    The query string of /api/v1/search contains the user's question (I-8).
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        scrub_url_attributes(span)

    def on_end(self, span: ReadableSpan) -> None:
        return None


def _one_line_json(span: ReadableSpan) -> str:
    return str(span.to_json(indent=None)) + "\n"


def build_tracer_provider(
    *, service: str, environment: str, exporter: Exporter = "none"
) -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({"service.name": service, "deployment.environment": environment})
    )
    provider.add_span_processor(ScrubbingSpanProcessor())  # BEFORE any exporter
    if exporter == "console":
        provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter(formatter=_one_line_json))
        )
    return provider
