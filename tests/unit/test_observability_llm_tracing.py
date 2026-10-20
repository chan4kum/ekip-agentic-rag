# ruff: noqa: S105, S106  # fake credentials in tests
import json
from collections.abc import Sequence

import pytest
from ekip_observability import (
    GenAI,
    NoopLLMTracer,
    OtelLLMTracer,
    add_otlp_export,
    langfuse_otlp_config,
)
from ekip_observability.llm_tracing import OtlpExporterConfig, truncate_redactor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def make(**kwargs: object) -> tuple[OtelLLMTracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OtelLLMTracer(provider, **kwargs), exporter  # type: ignore[arg-type]


def test_chat_span_carries_genai_metadata() -> None:
    tracer, exporter = make()
    with tracer.chat_span(model="gpt-x", provider="openai", temperature=0.0, max_tokens=300) as s:
        s.record_usage(input_tokens=120, output_tokens=30)
        s.record_response(model="gpt-x-2026", finish_reason="stop")
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-x"
    attrs = span.attributes or {}
    assert attrs[GenAI.OPERATION] == "chat"
    assert attrs[GenAI.PROVIDER] == "openai"
    assert attrs[GenAI.REQUEST_MODEL] == "gpt-x"
    assert attrs[GenAI.REQUEST_TEMPERATURE] == 0.0
    assert attrs[GenAI.REQUEST_MAX_TOKENS] == 300
    assert attrs[GenAI.INPUT_TOKENS] == 120
    assert attrs[GenAI.OUTPUT_TOKENS] == 30
    assert attrs[GenAI.RESPONSE_MODEL] == "gpt-x-2026"
    assert attrs[GenAI.FINISH_REASONS] == ("stop",)


def test_attribute_names_are_pinned() -> None:
    assert GenAI.OPERATION == "gen_ai.operation.name"
    assert GenAI.INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert GenAI.OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
    assert GenAI.REQUEST_MODEL == "gen_ai.request.model"


def test_content_is_not_captured_by_default() -> None:
    tracer, exporter = make()
    with tracer.chat_span(model="m", provider="p") as s:
        s.record_content(
            prompt=[{"role": "user", "content": "salary of Jane Doe"}], completion="it is 100"
        )
    attrs = exporter.get_finished_spans()[0].attributes or {}
    assert GenAI.INPUT_MESSAGES not in attrs
    assert GenAI.OUTPUT_MESSAGES not in attrs
    assert "Jane" not in json.dumps(dict(attrs), default=str)


def test_content_capture_goes_through_the_redactor() -> None:
    tracer, exporter = make(capture_content=True, redact=lambda t: t.replace("Jane", "[NAME]"))
    with tracer.chat_span(model="m", provider="p") as s:
        s.record_content(prompt=[{"role": "user", "content": "Jane Doe?"}], completion="Jane is")
    attrs = exporter.get_finished_spans()[0].attributes or {}
    sent = json.loads(str(attrs[GenAI.INPUT_MESSAGES]))
    assert sent == [{"role": "user", "content": "[NAME] Doe?"}]
    assert "Jane" not in str(attrs[GenAI.OUTPUT_MESSAGES])


def test_default_redactor_truncates() -> None:
    assert truncate_redactor(5)("abcdefghij") == "abcde...[truncated]"
    assert truncate_redactor(50)("short") == "short"


def test_exception_marks_the_span_and_propagates() -> None:
    tracer, exporter = make()
    with pytest.raises(TimeoutError), tracer.chat_span(model="m", provider="p"):
        raise TimeoutError("boom with secret text")
    span = exporter.get_finished_spans()[0]
    assert (span.attributes or {})[GenAI.ERROR_TYPE] == "TimeoutError"
    assert span.status.status_code.name == "ERROR"
    assert "secret text" not in json.dumps(dict(span.attributes or {}), default=str)


def test_operation_spans_validate_the_operation() -> None:
    tracer, exporter = make()
    with tracer.operation_span("retrieval", name="hybrid"):
        pass
    assert exporter.get_finished_spans()[0].name == "retrieval hybrid"
    with pytest.raises(ValueError):
        tracer.operation_span("made_up")


def test_noop_tracer_accepts_every_call() -> None:
    tracer = NoopLLMTracer()
    with tracer.chat_span(model="m", provider="p") as s:
        s.record_usage(input_tokens=1, output_tokens=1)
        s.record_content(prompt=[], completion="x")
        s.record_error("X")
    with tracer.operation_span("execute_tool"):
        pass


def test_langfuse_config_builds_the_otlp_endpoint_and_basic_auth() -> None:
    cfg = langfuse_otlp_config(
        base_url="https://cloud.langfuse.com/", public_key="pk-lf-1", secret_key="sk-lf-2"
    )
    assert cfg.endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert cfg.headers["Authorization"].startswith("Basic ")
    # base64("pk-lf-1:sk-lf-2")
    assert cfg.headers["Authorization"] == "Basic cGstbGYtMTpzay1sZi0y"


def test_exporter_config_never_prints_credentials() -> None:
    cfg = langfuse_otlp_config(base_url="https://h.example", public_key="pk", secret_key="sk-top")
    assert "sk-top" not in repr(cfg)
    assert "Basic" not in repr(cfg)


@pytest.mark.parametrize(
    ("url", "pk", "sk"),
    [("http://evil.example", "a", "b"), ("https://h", "", "b"), ("https://h", "a", "")],
)
def test_langfuse_config_rejects_bad_input(url: str, pk: str, sk: str) -> None:
    with pytest.raises(ValueError):
        langfuse_otlp_config(base_url=url, public_key=pk, secret_key=sk)


def test_localhost_http_is_allowed_for_a_local_langfuse() -> None:
    cfg = langfuse_otlp_config(base_url="http://localhost:3000", public_key="a", secret_key="b")
    assert cfg.endpoint.startswith("http://localhost:3000/")


class FakeExporter(SpanExporter):
    def __init__(self) -> None:
        self.names: list[str] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.names.extend(s.name for s in spans)
        return SpanExportResult.SUCCESS


def test_add_otlp_export_uses_the_injected_factory() -> None:
    fake = FakeExporter()
    seen: list[OtlpExporterConfig] = []

    def factory(cfg: OtlpExporterConfig) -> SpanExporter:
        seen.append(cfg)
        return fake

    provider = TracerProvider()
    cfg = langfuse_otlp_config(base_url="https://h.example", public_key="a", secret_key="b")
    add_otlp_export(provider, cfg, exporter_factory=factory)
    tracer = OtelLLMTracer(provider)
    with tracer.chat_span(model="m", provider="p"):
        pass
    provider.force_flush()
    provider.shutdown()
    assert seen == [cfg]
    assert fake.names == ["chat m"]
