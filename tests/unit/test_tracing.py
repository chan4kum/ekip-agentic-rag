import io
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import structlog
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_observability.logging import configure_logging, get_logger
from ekip_observability.tracing import _one_line_json, build_tracer_provider
from idp_testkit import TestIdp, open_limits
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
IDP = TestIdp()


def settings() -> Settings:
    return Settings(_env_file=None, app_env="test")


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    p = build_tracer_provider(service="ekip-api", environment="test")
    p.add_span_processor(SimpleSpanProcessor(exporter))  # added AFTER the scrubber
    return p


@pytest.fixture
def log_buf() -> Iterator[io.StringIO]:
    structlog.reset_defaults()
    stream = io.StringIO()
    configure_logging("INFO", json_logs=True, stream=stream)
    yield stream


@pytest.fixture
async def client(
    provider: TracerProvider, log_buf: io.StringIO
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        settings(),
        tracer_provider=provider,
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def server_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.kind.name == "SERVER"]


def log_lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


async def test_a_search_request_produces_one_server_span(
    client: httpx.AsyncClient, exporter: InMemorySpanExporter
) -> None:
    r = await client.get("/api/v1/search", params={"q": "torque"}, headers=IDP.auth())
    assert r.status_code == 200
    (span,) = server_spans(exporter)
    assert "/api/v1/search" in span.name
    attrs = span.attributes or {}
    assert 200 in (attrs.get("http.status_code"), attrs.get("http.response.status_code"))


async def test_the_users_question_never_appears_in_span_attributes(
    client: httpx.AsyncClient, exporter: InMemorySpanExporter
) -> None:
    await client.get("/api/v1/search", params={"q": "private question text"}, headers=IDP.auth())
    for span in exporter.get_finished_spans():
        dumped = json.dumps(dict(span.attributes or {}))
        assert "private" not in dumped, f"query leaked into span {span.name}: {dumped}"


async def test_probe_endpoints_are_not_traced(
    client: httpx.AsyncClient, exporter: InMemorySpanExporter
) -> None:
    await client.get("/health")
    await client.get("/ready")
    assert exporter.get_finished_spans() == ()


async def test_access_log_carries_the_same_trace_and_span_id_as_the_span(
    client: httpx.AsyncClient, exporter: InMemorySpanExporter, log_buf: io.StringIO
) -> None:
    await client.get("/api/v1/search", params={"q": "torque"}, headers=IDP.auth())
    (span,) = server_spans(exporter)
    ctx = span.get_span_context()
    assert ctx is not None
    access = next(r for r in log_lines(log_buf) if r["event"] == "http_request")
    assert access["trace_id"] == format(ctx.trace_id, "032x")
    assert access["span_id"] == format(ctx.span_id, "016x")


def test_logs_outside_a_request_have_no_trace_fields(log_buf: io.StringIO) -> None:
    get_logger("x").info("background work")
    (rec,) = log_lines(log_buf)
    assert "trace_id" not in rec and "span_id" not in rec


async def test_an_incoming_traceparent_is_continued(
    client: httpx.AsyncClient, exporter: InMemorySpanExporter
) -> None:
    await client.get(
        "/api/v1/search",
        params={"q": "torque"},
        headers={"traceparent": f"00-{TRACE_ID}-b7ad6b7169203331-01"},
    )
    (span,) = server_spans(exporter)
    ctx = span.get_span_context()
    assert ctx is not None
    assert format(ctx.trace_id, "032x") == TRACE_ID


def test_resource_identifies_the_service(provider: TracerProvider) -> None:
    attrs = provider.resource.attributes
    assert attrs["service.name"] == "ekip-api"
    assert attrs["deployment.environment"] == "test"


async def test_console_span_format_is_one_json_line(
    client: httpx.AsyncClient, exporter: InMemorySpanExporter
) -> None:
    await client.get("/api/v1/search", params={"q": "torque"}, headers=IDP.auth())
    line = _one_line_json(server_spans(exporter)[0])
    assert line.endswith("\n") and line.count("\n") == 1
    assert json.loads(line)["name"]


async def test_app_shuts_down_a_provider_it_built_but_not_one_it_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    built = build_tracer_provider(service="s", environment="test")
    monkeypatch.setattr(built, "shutdown", lambda: calls.append("built"))
    monkeypatch.setattr("ekip_api.main.build_tracer_provider", lambda **_: built)

    owned = create_app(settings())
    async with owned.router.lifespan_context(owned):
        pass
    assert calls == ["built"]

    injected = build_tracer_provider(service="s", environment="test")
    monkeypatch.setattr(injected, "shutdown", lambda: calls.append("injected"))
    app = create_app(settings(), tracer_provider=injected)
    async with app.router.lifespan_context(app):
        pass
    assert calls == ["built"]  # the injected provider is its owner's responsibility


def test_tracing_exporter_setting_defaults_to_none_and_rejects_unknown_values() -> None:
    assert settings().tracing_exporter == "none"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tracing_exporter="jaeger")


def test_scrubbing_processor_also_cleans_attributes_present_at_span_start(
    exporter: InMemorySpanExporter, provider: TracerProvider
) -> None:
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span(
        "GET https://api.example.com", attributes={"url.full": "https://x.test/v1?key=SECRET-KEY"}
    ):
        pass
    (span,) = exporter.get_finished_spans()
    assert (span.attributes or {})["url.full"] == "https://x.test/v1"


async def test_one_request_is_exactly_one_span(
    client: httpx.AsyncClient, exporter: InMemorySpanExporter
) -> None:
    await client.get("/api/v1/search", params={"q": "torque"}, headers=IDP.auth())
    assert len(exporter.get_finished_spans()) == 1  # no internal receive/send spans
