import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from script_loader import load_script

lt = load_script("load_test")


def make_app() -> FastAPI:
    app = FastAPI()
    state = {"n": 0}

    @app.post("/api/v1/chat")
    async def chat(request: Request) -> Response:
        auth = request.headers.get("authorization")
        if auth != "Bearer good":
            return JSONResponse({"detail": "no"}, status_code=401)
        body = await request.json()
        state["n"] += 1
        if body["message"] == "boom":
            return JSONResponse({"detail": "down"}, status_code=503)

        async def gen() -> AsyncIterator[bytes]:
            yield b"event: meta\ndata: {}\n\n"
            await asyncio.sleep(0.02)
            yield b"event: token\ndata: {}\n\n"
            await asyncio.sleep(0.02)
            yield b"event: done\ndata: {}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/sse-error")
    async def sse_error() -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            yield b"event: error\ndata: {}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app()), base_url="http://test")


async def test_measures_latency_and_time_to_first_token() -> None:
    async with client() as c:
        res = await lt.run_load(
            c,
            headers={"Authorization": "Bearer good"},
            concurrency=3,
            duration_s=None,
            max_requests=12,
        )
    m = lt.summarize(res)
    assert m["load.requests"] == 12
    assert m["load.error_rate"] == 0
    assert m["load.p95_ms"] >= 40  # two 20 ms sleeps
    assert 15 <= m["load.ttft_p50_ms"] < m["load.p50_ms"]
    assert m["load.rps"] > 0


async def test_server_errors_count_toward_error_rate_but_not_latency() -> None:
    async with client() as c:
        res = await lt.run_load(
            c,
            headers={"Authorization": "Bearer good"},
            messages=["ok", "boom"],
            concurrency=1,
            duration_s=None,
            max_requests=10,
        )
    m = lt.summarize(res)
    assert m["load.error_rate"] == pytest.approx(0.5)
    assert len(res.latencies_ms) == 5
    assert res.statuses["503"] == 5


async def test_auth_failures_are_rejections_not_server_errors() -> None:
    async with client() as c:
        res = await lt.run_load(c, concurrency=2, duration_s=None, max_requests=6)
    m = lt.summarize(res)
    assert m["load.error_rate"] == 0
    assert m["load.rejected_rate"] == 1.0
    assert "load.p95_ms" not in m  # no successful request, so no invented latency


async def test_in_stream_error_events_count_as_errors() -> None:
    async with client() as c:
        res = await lt.run_load(
            c, path="/sse-error", concurrency=1, duration_s=None, max_requests=3
        )
    assert lt.summarize(res)["load.error_rate"] == 1.0


async def test_transport_failures_are_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t") as c:
        res = await lt.run_load(c, concurrency=1, duration_s=None, max_requests=4)
    assert lt.summarize(res)["load.error_rate"] == 1.0


async def test_duration_mode_stops() -> None:
    async with client() as c:
        res = await lt.run_load(
            c, headers={"Authorization": "Bearer good"}, concurrency=2, duration_s=0.2
        )
    assert res.requests >= 2
    assert res.duration_s < 2


async def test_needs_a_stop_condition_and_valid_concurrency() -> None:
    async with client() as c:
        with pytest.raises(ValueError):
            await lt.run_load(c, duration_s=None, max_requests=None)
        with pytest.raises(ValueError):
            await lt.run_load(c, concurrency=0)


def test_refuses_remote_targets_unless_acknowledged() -> None:
    lt.ensure_target_allowed("http://localhost:8000", i_know=False)
    lt.ensure_target_allowed("http://127.0.0.1:8000", i_know=False)
    with pytest.raises(lt.TargetRefused):
        lt.ensure_target_allowed("https://api.example.com", i_know=False)
    with pytest.raises(lt.TargetRefused):
        lt.ensure_target_allowed("http://localhost.evil.example", i_know=False)
    lt.ensure_target_allowed("https://api.example.com", i_know=True)


def test_cli_refuses_remote_without_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["load_test", "--base-url", "https://prod.example.com"])
    assert lt.main() == 2
    assert "--i-know" in capsys.readouterr().err


def test_token_never_appears_in_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("EKIP_LOAD_TOKEN", "super-secret-token")
    out = tmp_path / "r.json"

    class Dummy:
        pass

    async def fake_run(client: object, **kw: Any) -> Any:
        assert kw["headers"] == {"Authorization": "Bearer super-secret-token"}
        res = lt.LoadResult(started=0.0, finished=1.0)
        res.requests = 1
        res.latencies_ms = [10.0]
        return res

    monkeypatch.setattr(lt, "run_load", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "load_test",
            "--base-url",
            "http://localhost:1",
            "--max-requests",
            "1",
            "--json-out",
            str(out),
        ],
    )
    assert lt.main() == 0
    captured = capsys.readouterr()
    assert "super-secret-token" not in captured.out + captured.err
    assert "super-secret-token" not in out.read_text()
    assert json.loads(out.read_text())["load.p95_ms"] == 10.0


def test_summary_of_an_empty_run_is_an_error_not_a_pass() -> None:
    m = lt.summarize(lt.LoadResult(started=0.0, finished=1.0))
    assert m["load.error_rate"] == 1.0
