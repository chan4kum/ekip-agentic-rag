"""One behaviour spec for every EmbeddingCache (memory and Redis)."""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from ekip_llm_gateway import EmbeddingCache, InMemoryEmbeddingCache
from ekip_persistence import RedisEmbeddingCache
from redis.asyncio import Redis

LIVE = os.environ.get("RUN_INTEGRATION") == "1"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture(params=["memory", pytest.param("redis", marks=pytest.mark.integration)])
async def cache(request: pytest.FixtureRequest) -> AsyncIterator[EmbeddingCache]:
    if request.param == "memory":
        yield InMemoryEmbeddingCache()
        return
    if not LIVE:
        pytest.skip("needs live Redis (RUN_INTEGRATION=1)")
    client = Redis.from_url(REDIS_URL)
    try:
        yield RedisEmbeddingCache(client, prefix=f"test:{uuid.uuid4().hex}:", ttl_s=60)
    finally:
        await client.aclose()


def k(name: str) -> str:
    return f"emb:t:m@3:document:{name}"


async def test_missing_keys_are_omitted(cache: EmbeddingCache) -> None:
    assert await cache.get_many([k("nope")]) == {}
    assert await cache.get_many([]) == {}


async def test_round_trip_preserves_order_and_values(cache: EmbeddingCache) -> None:
    await cache.set_many({k("a"): (0.25, -0.5, 0.125), k("b"): (1.0, 0.0, 0.0)})
    got = await cache.get_many([k("b"), k("a"), k("zzz")])
    assert set(got) == {k("a"), k("b")}
    assert got[k("a")] == pytest.approx((0.25, -0.5, 0.125), abs=1e-6)  # float32 storage
    assert got[k("b")] == pytest.approx((1.0, 0.0, 0.0))


async def test_overwrite_replaces_the_value(cache: EmbeddingCache) -> None:
    await cache.set_many({k("a"): (1.0, 0.0, 0.0)})
    await cache.set_many({k("a"): (0.0, 1.0, 0.0)})
    assert (await cache.get_many([k("a")]))[k("a")] == pytest.approx((0.0, 1.0, 0.0))


async def test_many_keys_in_one_call(cache: EmbeddingCache) -> None:
    items = {k(str(i)): (float(i), 0.5, 0.25) for i in range(500)}
    await cache.set_many(items)
    got = await cache.get_many(list(items))
    assert len(got) == 500 and got[k("499")][0] == pytest.approx(499.0)
