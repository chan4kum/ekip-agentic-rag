"""LLM-call tracing: a small port, an OpenTelemetry implementation, and Langfuse over OTLP.

Attribute names follow the OpenTelemetry GenAI semantic conventions. Those conventions are still
evolving, so every name lives in `GenAI` below (one place to review on upgrade) and the
opentelemetry packages are pinned in pyproject.

Privacy: prompt / completion CONTENT is not recorded unless `capture_content=True`, and even then
it passes through a redaction callable first (default: a hard length cap). Metadata (model, token
counts, finish reason) is always safe to record. Langfuse is reached through standard OTLP/HTTP -
no vendor SDK.
"""

import base64
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode


class GenAI:
    """Pinned attribute names (OpenTelemetry GenAI semantic conventions)."""

    OPERATION = "gen_ai.operation.name"
    PROVIDER = "gen_ai.provider.name"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    RESPONSE_MODEL = "gen_ai.response.model"
    FINISH_REASONS = "gen_ai.response.finish_reasons"
    INPUT_TOKENS = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    TOOL_NAME = "gen_ai.tool.name"
    ERROR_TYPE = "error.type"


OPERATIONS = frozenset({"chat", "embeddings", "retrieval", "execute_tool", "invoke_agent"})
MAX_CONTENT_CHARS = 2000

Redactor = Callable[[str], str]


def truncate_redactor(limit: int = MAX_CONTENT_CHARS) -> Redactor:
    def redact(text: str) -> str:
        return text if len(text) <= limit else text[:limit] + "...[truncated]"

    return redact


class LLMSpan(Protocol):
    def record_response(
        self, *, model: str | None = None, finish_reason: str | None = None
    ) -> None: ...

    def record_usage(self, *, input_tokens: int, output_tokens: int) -> None: ...

    def record_content(
        self, *, prompt: Sequence[Mapping[str, str]] | None, completion: str | None
    ) -> None: ...

    def record_error(self, error_type: str) -> None: ...


class LLMTracer(Protocol):
    def chat_span(
        self,
        *,
        model: str,
        provider: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AbstractContextManager[LLMSpan]: ...

    def operation_span(
        self, operation: str, *, name: str | None = None
    ) -> AbstractContextManager[LLMSpan]: ...


class _NoopSpan:
    def record_response(
        self, *, model: str | None = None, finish_reason: str | None = None
    ) -> None:
        return None

    def record_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        return None

    def record_content(
        self, *, prompt: Sequence[Mapping[str, str]] | None, completion: str | None
    ) -> None:
        return None

    def record_error(self, error_type: str) -> None:
        return None


class NoopLLMTracer:
    """Default when tracing is off: zero overhead, zero data."""

    @contextmanager
    def chat_span(
        self,
        *,
        model: str,
        provider: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[LLMSpan]:
        yield _NoopSpan()

    @contextmanager
    def operation_span(self, operation: str, *, name: str | None = None) -> Iterator[LLMSpan]:
        yield _NoopSpan()


class _OtelSpan:
    def __init__(self, span: Span, *, capture_content: bool, redact: Redactor) -> None:
        self._span = span
        self._capture = capture_content
        self._redact = redact

    def record_response(
        self, *, model: str | None = None, finish_reason: str | None = None
    ) -> None:
        if model:
            self._span.set_attribute(GenAI.RESPONSE_MODEL, model)
        if finish_reason:
            self._span.set_attribute(GenAI.FINISH_REASONS, [finish_reason])

    def record_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        self._span.set_attribute(GenAI.INPUT_TOKENS, input_tokens)
        self._span.set_attribute(GenAI.OUTPUT_TOKENS, output_tokens)

    def record_content(
        self, *, prompt: Sequence[Mapping[str, str]] | None, completion: str | None
    ) -> None:
        if not self._capture:
            return
        if prompt is not None:
            messages = [
                {"role": m.get("role", ""), "content": self._redact(m.get("content", ""))}
                for m in prompt
            ]
            self._span.set_attribute(GenAI.INPUT_MESSAGES, json.dumps(messages))
        if completion is not None:
            output = [{"role": "assistant", "content": self._redact(completion)}]
            self._span.set_attribute(GenAI.OUTPUT_MESSAGES, json.dumps(output))

    def record_error(self, error_type: str) -> None:
        self._span.set_attribute(GenAI.ERROR_TYPE, error_type)
        self._span.set_status(Status(StatusCode.ERROR))


class OtelLLMTracer:
    """OpenTelemetry implementation. The TracerProvider is injected (no globals)."""

    def __init__(
        self,
        provider: TracerProvider,
        *,
        capture_content: bool = False,
        redact: Redactor | None = None,
    ) -> None:
        self._tracer = provider.get_tracer("ekip.llm")
        self._capture = capture_content
        self._redact = redact or truncate_redactor()

    @contextmanager
    def _span(self, name: str, attributes: dict[str, Any]) -> Iterator[LLMSpan]:
        with self._tracer.start_as_current_span(name, attributes=attributes) as span:
            wrapper = _OtelSpan(span, capture_content=self._capture, redact=self._redact)
            try:
                yield wrapper
            except Exception as exc:
                wrapper.record_error(type(exc).__name__)
                raise

    def chat_span(
        self,
        *,
        model: str,
        provider: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AbstractContextManager[LLMSpan]:
        attributes: dict[str, Any] = {
            GenAI.OPERATION: "chat",
            GenAI.PROVIDER: provider,
            GenAI.REQUEST_MODEL: model,
        }
        if temperature is not None:
            attributes[GenAI.REQUEST_TEMPERATURE] = temperature
        if max_tokens is not None:
            attributes[GenAI.REQUEST_MAX_TOKENS] = max_tokens
        return self._span(f"chat {model}", attributes)

    def operation_span(
        self, operation: str, *, name: str | None = None
    ) -> AbstractContextManager[LLMSpan]:
        if operation not in OPERATIONS:
            raise ValueError(f"unknown GenAI operation: {operation}")
        label = f"{operation} {name}" if name else operation
        return self._span(label, {GenAI.OPERATION: operation})


# ---- Langfuse over OTLP/HTTP (no SDK) -------------------------------------------------------


@dataclass(frozen=True)
class OtlpExporterConfig:
    endpoint: str
    headers: Mapping[str, str] = field(repr=False)  # contains credentials: never printed
    timeout_seconds: float = 10.0


def langfuse_otlp_config(*, base_url: str, public_key: str, secret_key: str) -> OtlpExporterConfig:
    """Langfuse accepts OTLP traces at <host>/api/public/otel/v1/traces with Basic auth."""
    if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise ValueError("Langfuse base URL must be https (http only for localhost)")
    if not public_key or not secret_key:
        raise ValueError("Langfuse public and secret keys are both required")
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return OtlpExporterConfig(
        endpoint=f"{base_url.rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {token}", "x-langfuse-ingestion-version": "4"},
    )


ExporterFactory = Callable[[OtlpExporterConfig], SpanExporter]


def default_exporter_factory(config: OtlpExporterConfig) -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(
        endpoint=config.endpoint, headers=dict(config.headers), timeout=int(config.timeout_seconds)
    )


def add_otlp_export(
    provider: TracerProvider,
    config: OtlpExporterConfig,
    *,
    exporter_factory: ExporterFactory = default_exporter_factory,
) -> None:
    """Attach a batching OTLP exporter. Add AFTER the scrubbing processor (build_tracer_provider
    already registers that first), so nothing leaves the process unscrubbed."""
    provider.add_span_processor(BatchSpanProcessor(exporter_factory(config)))
