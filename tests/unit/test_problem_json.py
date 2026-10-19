import io
import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import structlog
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_observability.context import clear_request_context
from ekip_observability.logging import configure_logging
from fastapi import FastAPI
from idp_testkit import TestIdp, open_limits

IDP = TestIdp()


def build_app() -> FastAPI:
    app = create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
    )

    async def boom() -> None:
        raise RuntimeError("db password=hunter2 at 10.0.0.5")

    app.add_api_route("/boom", boom)
    return app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=build_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def log_buf() -> Iterator[io.StringIO]:
    structlog.reset_defaults()
    clear_request_context()
    stream = io.StringIO()
    configure_logging("INFO", json_logs=True, stream=stream)
    yield stream
    clear_request_context()
    structlog.reset_defaults()


def assert_problem(r: httpx.Response, status: int, slug: str) -> dict[str, object]:
    assert r.status_code == status
    assert r.headers["content-type"] == "application/problem+json"
    body: dict[str, object] = r.json()
    assert body["status"] == status
    assert body["type"] == f"urn:ekip:problem:{slug}"
    assert isinstance(body["title"], str) and isinstance(body["detail"], str)
    assert body["request_id"] == r.headers["x-request-id"]  # the id the caller can quote
    return body


async def test_401_is_problem_json_and_keeps_its_challenge_header(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/api/v1/search", params={"q": "x"})
    body = assert_problem(r, 401, "unauthorized")
    assert body["detail"] == "authentication required"
    assert r.headers["www-authenticate"] == "Bearer"


async def test_403_and_429_are_problem_json() -> None:
    from ekip_contracts.ratelimit import RateLimits
    from ekip_ratelimit.memory import InMemoryRateLimiter

    tight = RateLimits(
        user=InMemoryRateLimiter(capacity=1, refill_per_second=0.001),
        tenant=InMemoryRateLimiter(capacity=100, refill_per_second=1),
    )
    app = create_app(
        Settings(_env_file=None, app_env="test"), token_validator=IDP.validator(), rate_limits=tight
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        forbidden = await c.get("/api/v1/search", params={"q": "x"}, headers=IDP.auth(roles=[]))
        await c.get("/api/v1/search", params={"q": "x"}, headers=IDP.auth())
        limited = await c.get("/api/v1/search", params={"q": "x"}, headers=IDP.auth())
    assert_problem(forbidden, 403, "forbidden")
    assert_problem(limited, 429, "rate-limited")
    assert int(limited.headers["retry-after"]) >= 1


async def test_unknown_path_and_wrong_method(client: httpx.AsyncClient) -> None:
    assert_problem(await client.get("/nope"), 404, "not-found")
    assert_problem(await client.post("/health"), 405, "method-not-allowed")


async def test_validation_error_lists_where_but_never_echoes_the_input(
    client: httpx.AsyncClient,
) -> None:
    secret = "secret-" + "a" * 1100  # longer than the 1000-char query limit
    r = await client.get("/api/v1/search", params={"q": secret}, headers=IDP.auth())
    body = assert_problem(r, 422, "validation-error")
    errors = body["errors"]
    assert isinstance(errors, list) and errors[0]["loc"] == ["query", "q"]
    assert set(errors[0]) == {"loc", "msg", "type"}  # no `input`, no `ctx`
    assert "secret-" not in r.text


async def test_missing_parameter_is_422(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/search", headers=IDP.auth())
    assert_problem(r, 422, "validation-error")


async def test_unhandled_exception_is_a_generic_500_that_still_carries_the_request_id(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/boom", headers={"X-Request-ID": "trace-me-123"})
    body = assert_problem(r, 500, "internal-error")
    assert body["detail"] == "internal error"
    assert r.headers["x-request-id"] == "trace-me-123"
    assert "hunter2" not in r.text and "10.0.0.5" not in r.text and "RuntimeError" not in r.text


async def test_unhandled_exception_is_logged_with_the_request_id(
    client: httpx.AsyncClient, log_buf: io.StringIO
) -> None:
    await client.get("/boom", headers={"X-Request-ID": "trace-me-123"})
    lines = [json.loads(x) for x in log_buf.getvalue().splitlines() if x.strip()]
    err = next(x for x in lines if x["event"] == "unhandled_exception")
    assert err["request_id"] == "trace-me-123"
    assert err["level"] == "error"
    access = next(x for x in lines if x["event"] == "http_request")
    assert access["status"] == 500 and access["request_id"] == "trace-me-123"
