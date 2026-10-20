"""One behaviour spec for every CacheStore (memory and Redis)."""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from ekip_caching import CacheStore, MemoryCacheStore, RedisCacheStore
from redis.asyncio import Redis

LIVE = os.environ.get("RUN_INTEGRATION") == "1"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture(params=["memory", pytest.param("redis", marks=pytest.mark.integration)])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[CacheStore]:
    if request.param == "memory":
        yield MemoryCacheStore()
        return
    if not LIVE:
        pytest.skip("needs live Redis (RUN_INTEGRATION=1)")
    client = Redis.from_url(REDIS_URL)
    try:
        yield RedisCacheStore(client)
    finally:
        await client.aclose()


def k(name: str) -> str:
    return f"test:{uuid.uuid4().hex}:{name}"


async def test_round_trip_and_missing(store: CacheStore) -> None:
    key = k("a")
    assert await store.get(key) is None
    await store.set(key, b"hello", ttl_s=60)
    assert await store.get(key) == b"hello"
    await store.set(key, b"world", ttl_s=60)
    assert await store.get(key) == b"world"


async def test_delete(store: CacheStore) -> None:
    key = k("a")
    await store.set(key, b"x", ttl_s=60)
    await store.delete(key)
    await store.delete(key)  # idempotent
    assert await store.get(key) is None


async def test_delete_tag_removes_every_tagged_key_only(store: CacheStore) -> None:
    tag, other = k("tag"), k("other")
    a, b, c = k("a"), k("b"), k("c")
    await store.set(a, b"1", ttl_s=60, tags=[tag])
    await store.set(b, b"2", ttl_s=60, tags=[tag, other])
    await store.set(c, b"3", ttl_s=60, tags=[other])
    assert await store.delete_tag(tag) == 2
    assert (await store.get(a), await store.get(b)) == (None, None)
    assert await store.get(c) == b"3"
    assert await store.delete_tag(tag) == 0
    assert await store.delete_tag(k("never-used")) == 0


async def test_counter_init_does_not_overwrite_and_incr_is_atomic(store: CacheStore) -> None:
    key = k("gen")
    assert await store.get_or_init_counter(key, 100) == 100
    assert await store.get_or_init_counter(key, 999) == 100
    results = await asyncio.gather(*(store.incr(key) for _ in range(50)))
    assert sorted(results) == list(range(101, 151))
    assert await store.get_or_init_counter(key, 5) == 150


async def test_ttl_expires_entries(store: CacheStore) -> None:
    if not isinstance(store, MemoryCacheStore):
        pytest.skip("real-time expiry is covered by unit tests with a fake clock")
    now = [0.0]
    short = MemoryCacheStore(clock=lambda: now[0])
    await short.set("x", b"1", ttl_s=5, tags=["t"])
    now[0] = 4.9
    assert await short.get("x") == b"1"
    now[0] = 5.0
    assert await short.get("x") is None
    assert await short.delete_tag("t") == 0  # expired entries leave no tag residue


async def test_memory_store_is_bounded() -> None:
    small = MemoryCacheStore(max_entries=3)
    for i in range(5):
        await small.set(f"k{i}", b"v", ttl_s=60)
    assert [await small.get(f"k{i}") for i in range(5)] == [None, None, b"v", b"v", b"v"]
