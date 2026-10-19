import io
import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import structlog
from ekip_api.main import create_app
from ekip_api.middleware import _request_id
from ekip_api.schemas import HealthResponse
from ekip_config.settings import Settings
from ekip_observability.context import clear_request_context
from ekip_observability.logging import configure_logging


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, app_env="test", **overrides)  # type: ignore[arg-type]


@pytest.fixture
def log_buf() -> Iterator[io.StringIO]:
    structlog.reset_defaults()
    clear_request_context()
    stream = io.StringIO()
    configure_logging("INFO", json_logs=True, stream=stream)
    yield stream
    clear_request_context()
    structlog.reset_defaults()


@pytest.fixture
async def client(log_buf: io.StringIO) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(make_settings()))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health_returns_schema(client: httpx.AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = HealthResponse.model_validate(r.json())
    assert body.status == "ok"
    assert body.environment == "test"  # injected settings are used (DI)


async def test_request_id_is_generated(client: httpx.AsyncClient) -> None:
    r = await client.get("/health")
    assert len(r.headers["x-request-id"]) == 32


async def test_valid_incoming_request_id_is_echoed(client: httpx.AsyncClient) -> None:
    r = await client.get("/health", headers={"X-Request-ID": "trace-abc_1.2"})
    assert r.headers["x-request-id"] == "trace-abc_1.2"


@pytest.mark.parametrize("bad", ["x" * 200, "has space", "semi;colon"])
async def test_malformed_request_id_is_replaced(client: httpx.AsyncClient, bad: str) -> None:
    r = await client.get("/health", headers={"X-Request-ID": bad})
    assert r.headers["x-request-id"] != bad
    assert len(r.headers["x-request-id"]) == 32


def test_newline_in_request_id_is_rejected() -> None:
    forged = "x\nfake_log_line"
    assert _request_id(forged) != forged


async def test_access_log_carries_request_id_and_status(
    client: httpx.AsyncClient, log_buf: io.StringIO
) -> None:
    await client.get("/health", headers={"X-Request-ID": "rid-1"})
    rec = json.loads(log_buf.getvalue().splitlines()[-1])
    assert rec["event"] == "http_request"
    assert rec["request_id"] == "rid-1"
    assert rec["status"] == 200
    assert rec["path"] == "/health"
    assert rec["client_ip"] == "127.0.0.1"


async def test_unknown_route_is_404_with_request_id(client: httpx.AsyncClient) -> None:
    r = await client.get("/nope")
    assert r.status_code == 404
    assert "x-request-id" in r.headers


async def test_docs_are_hidden_in_production() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        database_url="postgresql+asyncpg://u:p@db.internal:5432/db",
        redis_url="redis://cache.internal:6379/0",
        opensearch_url="https://search.internal:9200",
    )
    transport = httpx.ASGITransport(app=create_app(settings))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        assert (await c.get("/docs")).status_code == 404
        assert (await c.get("/openapi.json")).status_code == 404


async def test_lifespan_logs_startup_and_shutdown(capsys: pytest.CaptureFixture[str]) -> None:
    structlog.reset_defaults()
    app = create_app(make_settings())
    async with app.router.lifespan_context(app):
        pass
    out = capsys.readouterr().out
    assert "startup" in out
    assert "shutdown" in out
    structlog.reset_defaults()


async def test_access_log_never_contains_the_query_string(
    client: httpx.AsyncClient, log_buf: io.StringIO
) -> None:
    await client.get("/api/v1/search", params={"q": "private question text"})
    assert "private question text" not in log_buf.getvalue()
