from collections.abc import AsyncIterator

import httpx
import pytest
from ekip_observability import EkipMetrics, install_metrics
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse


def make_app(
    metrics: EkipMetrics,
    *,
    token: str | None = "s3cret",  # noqa: S107
    anonymous: bool = False,
) -> FastAPI:
    app = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    @app.get("/boom")
    async def boom() -> None:
        raise HTTPException(status_code=503, detail="down")

    @app.get("/limited")
    async def limited() -> None:
        raise HTTPException(status_code=429, detail="slow down")

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            yield b"event: token\ndata: {}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    install_metrics(app, metrics, token=token, allow_anonymous=anonymous)
    return app


def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def count(m: EkipMetrics, route: str, status: str, method: str = "GET") -> float:
    v = m.registry.get_sample_value(
        "ekip_http_requests_total", {"method": method, "route": route, "status": status}
    )
    return 0.0 if v is None else v


async def test_route_template_not_raw_path_is_the_label() -> None:
    m = EkipMetrics()
    async with client(make_app(m)) as c:
        for i in range(5):
            assert (await c.get(f"/items/user-{i}@corp.com")).status_code == 200
    assert count(m, "/items/{item_id}", "200") == 5
    text = "".join(str(s) for f in m.registry.collect() for s in f.samples)
    assert "corp.com" not in text


async def test_error_and_rate_limit_statuses_are_recorded() -> None:
    m = EkipMetrics()
    async with client(make_app(m)) as c:
        await c.get("/boom")
        await c.get("/limited")
    assert count(m, "/boom", "503") == 1
    assert count(m, "/limited", "429") == 1
    assert m.registry.get_sample_value("ekip_rate_limit_rejections_total") == 1


async def test_unknown_paths_share_one_series() -> None:
    m = EkipMetrics()
    async with client(make_app(m)) as c:
        for i in range(10):
            await c.get(f"/nope/{i}")
    assert count(m, "unmatched", "404") == 10


async def test_in_progress_returns_to_zero_and_streams_are_counted() -> None:
    m = EkipMetrics()
    async with client(make_app(m)) as c:
        r = await c.get("/stream")
        assert "event: token" in r.text
    assert m.registry.get_sample_value("ekip_http_requests_in_progress") == 0
    assert count(m, "/stream", "200") == 1


async def test_metrics_endpoint_requires_the_token() -> None:
    m = EkipMetrics()
    async with client(make_app(m)) as c:
        await c.get("/items/1")
        assert (await c.get("/metrics")).status_code == 404
        assert (
            await c.get("/metrics", headers={"Authorization": "Bearer wrong"})
        ).status_code == 404
        ok = await c.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert "ekip_http_requests_total" in ok.text
    assert ok.headers["cache-control"] == "no-store"


async def test_metrics_endpoint_is_closed_when_no_token_is_configured() -> None:
    m = EkipMetrics()
    async with client(make_app(m, token=None)) as c:
        assert (await c.get("/metrics")).status_code == 404
        assert (await c.get("/metrics", headers={"Authorization": "Bearer "})).status_code == 404


async def test_anonymous_must_be_explicit() -> None:
    m = EkipMetrics()
    async with client(make_app(m, token=None, anonymous=True)) as c:
        assert (await c.get("/metrics")).status_code == 200


async def test_metrics_endpoint_is_get_only() -> None:
    m = EkipMetrics()
    async with client(make_app(m)) as c:
        r = await c.post("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 404


async def test_scraping_does_not_count_itself() -> None:
    m = EkipMetrics()
    async with client(make_app(m)) as c:
        await c.get("/metrics", headers={"Authorization": "Bearer s3cret"})
    assert count(m, "unmatched", "404") == 0
    assert m.registry.get_sample_value("ekip_http_requests_total", {"route": "/metrics"}) is None


@pytest.mark.parametrize("scope_type", ["lifespan", "websocket"])
async def test_non_http_scopes_pass_through(scope_type: str) -> None:
    from ekip_observability.asgi import MetricsMiddleware

    seen: list[str] = []

    async def inner(scope, receive, send):  # type: ignore[no-untyped-def]
        seen.append(scope["type"])

    mw = MetricsMiddleware(inner, metrics=EkipMetrics())
    await mw({"type": scope_type}, None, None)  # type: ignore[arg-type]
    assert seen == [scope_type]
