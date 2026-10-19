from collections.abc import AsyncIterator

import httpx
import pytest
import yaml
from ekip_api.body_limit import MaxBodySizeMiddleware
from ekip_api.errors import install_error_handlers
from ekip_api.main import create_app
from ekip_api.middleware import RequestContextMiddleware
from ekip_config.settings import Settings
from fastapi import FastAPI, Request
from idp_testkit import TestIdp, open_limits
from tests_paths import ROOT

LIMIT = 1000


def mini_app() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)
    calls: list[int] = []

    @app.post("/upload")
    async def upload(request: Request) -> dict[str, int]:
        calls.append(1)
        return {"bytes": len(await request.body())}

    app.state.calls = calls
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=LIMIT)
    app.add_middleware(RequestContextMiddleware)
    return app


def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_body_within_the_limit_passes() -> None:
    async with client(mini_app()) as c:
        r = await c.post("/upload", content=b"x" * LIMIT)
    assert r.status_code == 200 and r.json() == {"bytes": LIMIT}


async def test_declared_oversize_is_rejected_before_the_handler_runs() -> None:
    app = mini_app()
    async with client(app) as c:
        r = await c.post("/upload", content=b"x" * (LIMIT + 1), headers={"X-Request-ID": "rid-9"})
    assert r.status_code == 413
    assert r.headers["content-type"] == "application/problem+json"
    assert r.json()["type"] == "urn:ekip:problem:payload-too-large"
    assert r.json()["request_id"] == "rid-9" == r.headers["x-request-id"]
    assert app.state.calls == []


async def test_chunked_upload_without_content_length_is_cut_off_at_the_limit() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(10):  # 10 x 200 bytes = 2000 > LIMIT, and no Content-Length is declared
            yield b"x" * 200

    async with client(mini_app()) as c:
        r = await c.post("/upload", content=chunks())
    assert r.status_code == 413
    assert r.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize("bad", ["abc", "-5", "1e3", "1 000"])
async def test_malformed_content_length_is_400(bad: str) -> None:
    app = mini_app()
    async with client(app) as c:
        r = await c.post("/upload", content=b"x", headers={"Content-Length": bad})
    assert r.status_code == 400
    assert app.state.calls == []


def test_invalid_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        MaxBodySizeMiddleware(lambda *a: None, max_bytes=0)  # type: ignore[arg-type]


async def test_production_app_enforces_the_configured_limit_on_chat() -> None:
    idp = TestIdp()
    app = create_app(
        Settings(_env_file=None, app_env="test", max_request_body_bytes=2048),
        token_validator=idp.validator(),
        rate_limits=open_limits(),
    )
    async with client(app) as c:
        big = await c.post("/api/v1/chat", json={"message": "a" * 3000}, headers=idp.auth())
        small = await c.post("/api/v1/chat", json={"message": "torque"}, headers=idp.auth())
    assert big.status_code == 413 and small.status_code == 200


def test_default_limit_matches_the_edge_limit() -> None:

    cfg = yaml.safe_load((ROOT / "infrastructure/traefik/dynamic/api.yml").read_text())
    edge = cfg["http"]["middlewares"]["body-limit"]["buffering"]["maxRequestBodyBytes"]
    assert Settings(_env_file=None).max_request_body_bytes == edge
