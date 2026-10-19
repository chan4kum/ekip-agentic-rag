"""Store semantics (memory always, Redis with RUN_INTEGRATION=1), the service, and an HTTP route."""

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

import httpx
import pytest
from ekip_api.adapters.in_memory_idempotency import InMemoryIdempotencyStore
from ekip_api.adapters.redis_idempotency import RedisIdempotencyStore
from ekip_api.dependencies import require_permission
from ekip_api.idempotency_http import get_idempotency, idempotency_key, run_idempotent
from ekip_api.main import create_app
from ekip_api.services.idempotency import (
    IdempotencyInProgressError,
    IdempotencyMismatchError,
    IdempotencyService,
    fingerprint_request,
)
from ekip_config.settings import Settings
from ekip_contracts.authorization import Action
from ekip_contracts.idempotency import IdempotencyStore, StoredResponse
from ekip_contracts.identity import SecurityContext
from fastapi import Depends, FastAPI, Request, Response
from idp_testkit import TestIdp, open_limits
from redis.asyncio import Redis

LIVE = os.environ.get("RUN_INTEGRATION") == "1"
OK = StoredResponse(200, b'{"done":true}', "application/json")


# ---- store semantics: one spec, every implementation ------------------------------------------
@pytest.fixture(
    params=[
        "memory",
        pytest.param(
            "redis",
            marks=[
                pytest.mark.integration,
                pytest.mark.skipif(not LIVE, reason="needs live Redis (RUN_INTEGRATION=1)"),
            ],
        ),
    ]
)
async def store(request: pytest.FixtureRequest) -> AsyncIterator[IdempotencyStore]:
    if request.param == "memory":
        yield InMemoryIdempotencyStore()
        return
    redis = Redis.from_url(str(Settings(_env_file=None).redis_url))
    prefix = f"test:idem:{uuid.uuid4().hex}"
    yield RedisIdempotencyStore(redis, prefix=prefix)
    async for key in redis.scan_iter(f"{prefix}:*"):
        await redis.delete(key)
    await redis.aclose()


async def test_first_begin_starts_and_second_sees_in_progress(store: IdempotencyStore) -> None:
    assert (await store.begin("k", "fp", lock_ttl_s=5)).kind == "started"
    assert (await store.begin("k", "fp", lock_ttl_s=5)).kind == "in_progress"


async def test_different_request_under_same_key_is_a_mismatch(store: IdempotencyStore) -> None:
    await store.begin("k", "fp-1", lock_ttl_s=5)
    assert (await store.begin("k", "fp-2", lock_ttl_s=5)).kind == "mismatch"
    await store.complete("k", "fp-1", OK, ttl_s=5)
    assert (await store.begin("k", "fp-2", lock_ttl_s=5)).kind == "mismatch"


async def test_completed_request_replays_the_exact_response(store: IdempotencyStore) -> None:
    await store.begin("k", "fp", lock_ttl_s=5)
    binary = StoredResponse(201, bytes(range(256)), "application/octet-stream")
    await store.complete("k", "fp", binary, ttl_s=5)
    replay = await store.begin("k", "fp", lock_ttl_s=5)
    assert replay.kind == "replay" and replay.response == binary


async def test_abandon_lets_the_client_retry(store: IdempotencyStore) -> None:
    await store.begin("k", "fp", lock_ttl_s=5)
    await store.abandon("k")
    assert (await store.begin("k", "fp", lock_ttl_s=5)).kind == "started"


async def test_concurrent_begins_have_exactly_one_winner(store: IdempotencyStore) -> None:
    results = await asyncio.gather(*(store.begin("k", "fp", lock_ttl_s=5) for _ in range(25)))
    assert sorted(r.kind for r in results).count("started") == 1


async def test_lock_expires_so_a_crashed_attempt_does_not_block_forever(
    store: IdempotencyStore,
) -> None:
    await store.begin("k", "fp", lock_ttl_s=0.1)
    await asyncio.sleep(0.2)
    assert (await store.begin("k", "fp", lock_ttl_s=5)).kind == "started"


async def test_memory_store_result_expiry_with_a_fake_clock() -> None:
    now = [0.0]
    s = InMemoryIdempotencyStore(clock=lambda: now[0])
    await s.begin("k", "fp", lock_ttl_s=5)
    await s.complete("k", "fp", OK, ttl_s=10)
    now[0] = 9
    assert (await s.begin("k", "fp", lock_ttl_s=5)).kind == "replay"
    now[0] = 11
    assert (await s.begin("k", "fp", lock_ttl_s=5)).kind == "started"


# ---- service ----------------------------------------------------------------------------------
def svc(max_body_bytes: int = 256 * 1024) -> IdempotencyService:
    return IdempotencyService(InMemoryIdempotencyStore(), max_body_bytes=max_body_bytes)


async def test_service_runs_the_operation_once_and_replays_after() -> None:
    s, calls = svc(), []

    async def produce() -> StoredResponse:
        calls.append(1)
        return OK

    first = await s.run(scope="t:u:POST:/x", key="key-12345", fingerprint="fp", produce=produce)
    second = await s.run(scope="t:u:POST:/x", key="key-12345", fingerprint="fp", produce=produce)
    assert first == (OK, False) and second == (OK, True)
    assert len(calls) == 1


async def test_same_key_in_another_scope_is_independent() -> None:
    s, calls = svc(), []

    async def produce() -> StoredResponse:
        calls.append(1)
        return OK

    await s.run(scope="tenant-a:u:POST:/x", key="key-12345", fingerprint="fp", produce=produce)
    await s.run(scope="tenant-b:u:POST:/x", key="key-12345", fingerprint="fp", produce=produce)
    assert len(calls) == 2


async def test_failure_releases_the_key_so_a_retry_runs_again() -> None:
    s, calls = svc(), []

    async def flaky() -> StoredResponse:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return OK

    with pytest.raises(RuntimeError):
        await s.run(scope="s", key="key-12345", fingerprint="fp", produce=flaky)
    assert (await s.run(scope="s", key="key-12345", fingerprint="fp", produce=flaky))[0] == OK


async def test_cancellation_releases_the_key_too() -> None:
    s = svc()

    async def hang() -> StoredResponse:
        await asyncio.sleep(10)
        return OK

    task = asyncio.create_task(s.run(scope="s", key="key-12345", fingerprint="fp", produce=hang))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async def quick() -> StoredResponse:
        return OK

    assert (await s.run(scope="s", key="key-12345", fingerprint="fp", produce=quick))[1] is False


async def test_server_errors_and_oversized_bodies_are_not_frozen() -> None:
    s, calls = svc(max_body_bytes=10), []

    async def error() -> StoredResponse:
        calls.append(1)
        return StoredResponse(503, b"{}", "application/json")

    async def huge() -> StoredResponse:
        calls.append(1)
        return StoredResponse(200, b"x" * 11, "text/plain")

    for produce, key in ((error, "key-error-1"), (huge, "key-huge-01")):
        await s.run(scope="s", key=key, fingerprint="fp", produce=produce)
        await s.run(scope="s", key=key, fingerprint="fp", produce=produce)
    assert len(calls) == 4  # each ran twice: nothing was cached


async def test_client_errors_are_cached_because_they_are_deterministic() -> None:
    s, calls = svc(), []

    async def bad() -> StoredResponse:
        calls.append(1)
        return StoredResponse(400, b"{}", "application/json")

    await s.run(scope="s", key="key-12345", fingerprint="fp", produce=bad)
    again = await s.run(scope="s", key="key-12345", fingerprint="fp", produce=bad)
    assert again[1] is True and len(calls) == 1


async def test_in_flight_duplicate_and_mismatch_raise_domain_errors() -> None:
    s = svc()
    gate = asyncio.Event()

    async def slow() -> StoredResponse:
        await gate.wait()
        return OK

    first = asyncio.create_task(s.run(scope="s", key="key-12345", fingerprint="fp", produce=slow))
    await asyncio.sleep(0.02)
    with pytest.raises(IdempotencyInProgressError):
        await s.run(scope="s", key="key-12345", fingerprint="fp", produce=slow)
    with pytest.raises(IdempotencyMismatchError):
        await s.run(scope="s", key="key-12345", fingerprint="other", produce=slow)
    gate.set()
    await first


def test_fingerprint_depends_on_method_path_and_body_with_unambiguous_boundaries() -> None:
    base = fingerprint_request("POST", "/x", b"a")
    assert base == fingerprint_request("post", "/x", b"a")
    assert base != fingerprint_request("POST", "/y", b"a")
    assert base != fingerprint_request("POST", "/x", b"b")
    assert fingerprint_request("POST", "/xa", b"") != fingerprint_request("POST", "/x", b"a")


# ---- HTTP route -------------------------------------------------------------------------------
IDP = TestIdp()
HITS: list[str] = []
GATE: dict[str, asyncio.Event] = {}


def build_app(service: IdempotencyService | None = None) -> FastAPI:
    app = create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
        idempotency=service or svc(),
    )

    @app.post("/api/v1/_echo")
    async def echo(
        request: Request,
        ctx: Annotated[SecurityContext, Depends(require_permission(Action.DOCUMENTS_READ))],
        key: Annotated[str, Depends(idempotency_key)],
        service: Annotated[IdempotencyService, Depends(get_idempotency)],
    ) -> Response:
        body = await request.body()

        async def produce() -> StoredResponse:
            HITS.append(body.decode())
            if "wait" in GATE:
                await GATE["wait"].wait()
            return StoredResponse(200, json.dumps({"n": len(HITS)}).encode(), "application/json")

        return await run_idempotent(
            service,
            scope=f"{ctx.tenant_id}:{ctx.subject}:POST:/api/v1/_echo",
            key=key,
            fingerprint=fingerprint_request("POST", "/api/v1/_echo", body),
            produce=produce,
        )

    return app


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    HITS.clear()
    GATE.clear()
    transport = httpx.ASGITransport(app=build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def post(
    c: httpx.AsyncClient, key: str | None, body: str = "a", **token: object
) -> httpx.Response:
    headers = {**IDP.auth(**token)}
    if key is not None:
        headers["Idempotency-Key"] = key
    return await c.post("/api/v1/_echo", content=body, headers=headers)


async def test_http_retry_replays_without_running_the_operation_again(
    client: httpx.AsyncClient,
) -> None:
    first = await post(client, "key-12345")
    second = await post(client, "key-12345")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == {"n": 1}
    assert "idempotent-replayed" not in first.headers
    assert second.headers["idempotent-replayed"] == "true"
    assert HITS == ["a"]


async def test_http_missing_or_invalid_key_is_400(client: httpx.AsyncClient) -> None:
    for key in (None, "short", "has space in it!", "x" * 200):
        r = await post(client, key)
        assert r.status_code == 400 and r.headers["content-type"] == "application/problem+json"
    assert HITS == []


async def test_http_same_key_different_body_is_422_with_its_own_problem_type(
    client: httpx.AsyncClient,
) -> None:
    await post(client, "key-12345", "a")
    r = await post(client, "key-12345", "different")
    assert r.status_code == 422
    assert r.json()["type"] == "urn:ekip:problem:idempotency-key-reused"
    assert HITS == ["a"]


async def test_http_keys_are_private_to_the_caller(client: httpx.AsyncClient) -> None:
    await post(client, "key-12345", sub="alice")
    other_user = await post(client, "key-12345", sub="bob")
    other_tenant = await post(client, "key-12345", sub="alice", tenant_id="tenant-other")
    assert "idempotent-replayed" not in other_user.headers
    assert "idempotent-replayed" not in other_tenant.headers
    assert len(HITS) == 3  # nobody could replay (or probe) someone else's result


async def test_http_concurrent_duplicate_gets_409_with_retry_after() -> None:
    HITS.clear()
    GATE["wait"] = asyncio.Event()
    transport = httpx.ASGITransport(app=build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        slow = asyncio.create_task(post(c, "key-12345"))
        await asyncio.sleep(0.05)
        dup = await post(c, "key-12345")
        GATE["wait"].set()
        done = await slow
    assert dup.status_code == 409
    assert dup.json()["type"] == "urn:ekip:problem:idempotency-in-progress"
    assert dup.headers["retry-after"] == "1"
    assert done.status_code == 200 and HITS == ["a"]


async def test_http_unauthenticated_request_never_touches_the_store(
    client: httpx.AsyncClient,
) -> None:
    r = await client.post("/api/v1/_echo", content="a", headers={"Idempotency-Key": "key-12345"})
    assert r.status_code == 401 and HITS == []
