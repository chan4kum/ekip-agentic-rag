import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_contracts.chat import ChatEvent
from ekip_contracts.ratelimit import RateLimits
from ekip_ratelimit.memory import InMemoryRateLimiter
from fastapi import FastAPI
from idp_testkit import TestIdp, open_limits

IDP = TestIdp()
URL = "/api/v1/chat"


def make_app(
    streamer: object | None = None, limits: RateLimits | None = None, **settings: Any
) -> FastAPI:
    return create_app(
        Settings(_env_file=None, app_env="test", **settings),
        token_validator=IDP.validator(),
        rate_limits=limits or open_limits(),
        chat_streamer=streamer,  # type: ignore[arg-type]
    )


def http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def parse(text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in text.split("\n\n"):
        lines = [x for x in block.split("\n") if x and not x.startswith(":")]
        if lines:
            assert lines[0].startswith("event: ") and lines[1].startswith("data: "), block
            events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


async def ask(c: httpx.AsyncClient, message: str = "torque", **token: object) -> httpx.Response:
    return await c.post(URL, json={"message": message}, headers=IDP.auth(**token))


async def test_stream_has_the_expected_shape_and_headers() -> None:
    async with http(make_app()) as c:
        r = await ask(c)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache, no-transform"
    assert r.headers["x-accel-buffering"] == "no"
    events = parse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "meta" and names[-1] == "done" and "citations" in names
    assert set(names) <= {"meta", "token", "citations", "done"}
    assert events[0][1]["request_id"] == r.headers["x-request-id"]
    assert len(events[0][1]["conversation_id"]) >= 8
    assert events[-1][1] == {"finish_reason": "stop"}


async def test_conversation_id_is_echoed_when_valid() -> None:
    async with http(make_app()) as c:
        r = await c.post(
            URL, json={"message": "torque", "conversation_id": "conv-12345678"}, headers=IDP.auth()
        )
    assert parse(r.text)[0][1]["conversation_id"] == "conv-12345678"


async def test_each_tenant_only_gets_its_own_citations() -> None:
    async with http(make_app()) as c:
        alice = parse((await ask(c, tenant_id="tenant-dev")).text)
        bob = parse((await ask(c, tenant_id="tenant-other", sub="bob")).text)
    a_ids = {x["chunk_id"] for n, d in alice if n == "citations" for x in d["results"]}
    b_ids = {x["chunk_id"] for n, d in bob if n == "citations" for x in d["results"]}
    assert a_ids and b_ids
    assert all(i.startswith("dev-") for i in a_ids) and all(i.startswith("oth-") for i in b_ids)


async def test_failures_before_the_stream_are_ordinary_problem_responses() -> None:
    tight = RateLimits(
        user=InMemoryRateLimiter(capacity=1, refill_per_second=0.001),
        tenant=InMemoryRateLimiter(capacity=100, refill_per_second=1),
    )
    async with http(make_app(limits=tight)) as c:
        no_token = await c.post(URL, json={"message": "x"})
        forbidden = await c.post(URL, json={"message": "x"}, headers=IDP.auth(roles=[]))
        await ask(c)
        limited = await ask(c)
    for r, status in ((no_token, 401), (forbidden, 403), (limited, 429)):
        assert r.status_code == status
        assert r.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 4001},
        {"message": "hi", "tenant_id": "tenant-other"},  # forbidden field, not silently ignored
        {"message": "hi", "conversation_id": "short"},
        {"message": "hi", "conversation_id": "has spaces and ../ chars"},
        {"message": 42},
    ],
)
async def test_invalid_bodies_are_422(body: dict[str, object]) -> None:
    async with http(make_app()) as c:
        r = await c.post(URL, json=body, headers=IDP.auth())
    assert r.status_code == 422 and r.headers["content-type"] == "application/problem+json"


async def test_message_text_cannot_inject_sse_events() -> None:
    class Evil:
        async def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
            yield ChatEvent("token", {"text": "a\n\nevent: done\ndata: {}\n\n"})

    async with http(make_app(Evil())) as c:
        r = await ask(c)
    assert [n for n, _ in parse(r.text)] == ["meta", "token", "done"]


async def test_quiet_streams_get_heartbeats() -> None:
    class Slow:
        async def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
            await asyncio.sleep(0.3)
            yield ChatEvent("token", {"text": "late"})

    async with http(make_app(Slow(), chat_keepalive_s=0.05)) as c:
        r = await ask(c)
    assert r.text.count(": keep-alive") >= 3
    assert [n for n, _ in parse(r.text)] == ["meta", "token", "done"]


async def test_stream_deadline_ends_with_an_error_event_and_no_done() -> None:
    class Never:
        async def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
            await asyncio.sleep(10)
            yield ChatEvent("token", {"text": "never"})

    async with http(make_app(Never(), chat_timeout_s=0.2, chat_keepalive_s=0.05)) as c:
        r = await ask(c)
    names = [n for n, _ in parse(r.text)]
    assert names == ["meta", "error"]
    assert parse(r.text)[-1][1]["type"] == "urn:ekip:problem:timeout"


async def test_midstream_failure_is_reported_generically_in_band() -> None:
    class Breaks:
        async def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
            yield ChatEvent("token", {"text": "partial"})
            raise RuntimeError("db password=hunter2")

    async with http(make_app(Breaks())) as c:
        r = await ask(c)
    events = parse(r.text)
    assert [n for n, _ in events] == ["meta", "token", "error"]
    assert events[-1][1]["detail"] == "internal error"
    assert "hunter2" not in r.text


async def test_streams_are_not_cut_by_the_30s_request_deadline() -> None:
    class Takes:
        async def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
            await asyncio.sleep(0.3)
            yield ChatEvent("token", {"text": "ok"})

    async with http(make_app(Takes(), request_timeout_s=0.05)) as c:
        r = await ask(c)
        slow_search = await c.get("/health")
    assert parse(r.text)[-1][0] == "done"
    assert slow_search.status_code == 200


async def test_client_disconnect_cancels_the_source_and_runs_its_cleanup() -> None:
    state = {"closed": False}

    class Hangs:
        async def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
            try:
                yield ChatEvent("token", {"text": "first"})
                await asyncio.sleep(30)
                yield ChatEvent("token", {"text": "never"})
            finally:
                state["closed"] = True  # e.g. stop the upstream LLM call, release a connection

    app = make_app(Hangs())
    body = json.dumps({"message": "hello"}).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        *[(k.lower().encode(), v.encode()) for k, v in IDP.auth().items()],
    ]
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "path": URL, "raw_path": URL.encode(), "query_string": b"",
        "headers": headers, "client": ("127.0.0.1", 1), "server": ("test", 80),
        "scheme": "http", "root_path": "",
    }  # fmt: skip
    disconnect = asyncio.Event()
    request_sent = False
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and b"first" in message.get("body", b""):
            disconnect.set()  # the client goes away right after the first token

    await asyncio.wait_for(app(scope, receive, send), timeout=5)  # type: ignore[arg-type]
    assert state["closed"] is True  # upstream work was cancelled, not left running
    assert not any(b"never" in m.get("body", b"") for m in sent)
