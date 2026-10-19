import asyncio
import io
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
import yaml
from ekip_api.deadline import RequestDeadlineMiddleware
from ekip_api.errors import install_error_handlers
from ekip_api.main import create_app
from ekip_api.middleware import RequestContextMiddleware
from ekip_config.settings import Settings
from ekip_observability.context import clear_request_context
from ekip_observability.logging import configure_logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from idp_testkit import TestIdp, open_limits

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def log_buf() -> Iterator[io.StringIO]:
    structlog.reset_defaults()
    clear_request_context()
    stream = io.StringIO()
    configure_logging("INFO", json_logs=True, stream=stream)
    yield stream
    clear_request_context()
    structlog.reset_defaults()


def mini_app(
    timeout_s: float = 0.1, exempt: tuple[str, ...] = ()
) -> tuple[FastAPI, dict[str, object]]:
    """Smallest app with the same middleware order as production."""
    state: dict[str, object] = {"cleaned_up": False, "cancelled": False}
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/fast")
    async def fast() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/slow")
    async def slow() -> dict[str, str]:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        finally:
            state["cleaned_up"] = True  # e.g. releasing a DB connection or lock
        return {"ok": "never"}

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            yield b"first-chunk"
            await asyncio.sleep(5)
            yield b"never"

        return StreamingResponse(gen())

    @app.get("/own-timeout")
    async def own_timeout() -> None:
        raise TimeoutError("handler's own failure")

    app.add_middleware(RequestDeadlineMiddleware, timeout_s=timeout_s, exempt_prefixes=exempt)
    app.add_middleware(RequestContextMiddleware)
    return app, state


def client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_fast_requests_are_unaffected() -> None:
    app, _ = mini_app()
    async with client(app) as c:
        assert (await c.get("/fast")).json() == {"ok": "yes"}


async def test_slow_handler_is_cancelled_and_answered_with_a_504_problem() -> None:
    app, state = mini_app(timeout_s=0.1)
    async with client(app) as c:
        started = asyncio.get_running_loop().time()
        r = await c.get("/slow", headers={"X-Request-ID": "rid-1"})
        elapsed = asyncio.get_running_loop().time() - started
    assert r.status_code == 504
    assert r.headers["content-type"] == "application/problem+json"
    assert r.json()["type"] == "urn:ekip:problem:timeout"
    assert r.json()["request_id"] == "rid-1" == r.headers["x-request-id"]
    assert elapsed < 1.0  # answered at the deadline, not after the 5 s sleep
    assert state["cancelled"] is True  # the handler really was cancelled...
    assert state["cleaned_up"] is True  # ...and its `finally` (resource release) ran


async def test_deadline_is_logged_and_the_access_log_shows_504(log_buf: io.StringIO) -> None:
    app, _ = mini_app(timeout_s=0.05)
    async with client(app) as c:
        await c.get("/slow")
    lines = [json.loads(x) for x in log_buf.getvalue().splitlines() if x.strip()]
    deadline = next(x for x in lines if x["event"] == "request_deadline_exceeded")
    assert deadline["path"] == "/slow" and deadline["response_started"] is False
    assert next(x for x in lines if x["event"] == "http_request")["status"] == 504


async def test_exempt_prefix_is_not_limited() -> None:
    app, _ = mini_app(timeout_s=0.05, exempt=("/fast",))
    async with client(app) as c:
        assert (await c.get("/fast")).status_code == 200


async def run_asgi(app: FastAPI, path: str) -> list[dict[str, Any]]:
    """Drive the app at the ASGI level, like a server would, and record what it sent."""
    sent: list[dict[str, Any]] = []
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "path": path, "raw_path": path.encode(), "query_string": b"", "headers": [],
        "client": ("127.0.0.1", 1), "server": ("test", 80), "scheme": "http", "root_path": "",
    }  # fmt: skip

    async def receive() -> dict[str, Any]:
        await asyncio.sleep(3600)  # no more request data, client stays connected
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)  # type: ignore[arg-type]
    return sent


async def test_a_stream_already_started_is_cut_not_converted_to_504(log_buf: io.StringIO) -> None:
    app, _ = mini_app(timeout_s=0.1)
    sent = await asyncio.wait_for(run_asgi(app, "/stream"), timeout=2)
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 200
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert bodies[0]["body"] == b"first-chunk"
    # The terminating message (more_body=False) is deliberately never sent: the server then drops
    # the connection, which is how a client can tell the answer was truncated.
    assert all(m.get("more_body", False) for m in bodies)
    assert '"response_started": true' in log_buf.getvalue()


async def test_a_timeout_raised_by_the_handler_is_not_mistaken_for_the_deadline() -> None:
    app, _ = mini_app(timeout_s=5)
    async with client(app) as c:
        r = await c.get("/own-timeout")
    assert r.status_code == 500  # an ordinary unhandled error, not a 504


def test_invalid_timeout_is_rejected() -> None:
    with pytest.raises(ValueError):
        RequestDeadlineMiddleware(lambda *a: None, timeout_s=0)  # type: ignore[arg-type]


def test_production_app_is_wired_with_the_configured_deadline() -> None:
    app = create_app(
        Settings(_env_file=None, app_env="test", request_timeout_s=12),
        token_validator=TestIdp().validator(),
        rate_limits=open_limits(),
    )
    deadline = next(m for m in app.user_middleware if "RequestDeadlineMiddleware" in repr(m.cls))
    assert deadline.kwargs["timeout_s"] == 12


def test_app_deadline_is_below_the_proxy_timeout() -> None:
    """If the app's deadline were longer than the proxy's, callers would see opaque proxy errors."""
    cfg = yaml.safe_load((ROOT / "infrastructure/traefik/dynamic/api.yml").read_text())
    header_timeout = cfg["http"]["serversTransports"]["api-transport"]["forwardingTimeouts"][
        "responseHeaderTimeout"
    ]
    proxy_s = float(str(header_timeout).rstrip("s"))
    assert Settings(_env_file=None).request_timeout_s < proxy_s
