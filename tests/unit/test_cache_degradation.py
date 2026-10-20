"""The cache is an optimisation: if the store is down, answers are still computed."""

from cache_testkit import VERSIONS, Counter, FailingStore, FakeVisibility, answer, hit, scope
from ekip_caching import (
    CacheInvalidator,
    CacheStore,
    CorpusGenerations,
    CounterCacheMetrics,
    RedisCacheStore,
    ResponseCache,
    RetrievalCache,
)
from ekip_contracts.retrieval import RetrievedChunk
from redis.asyncio import Redis


def response_cache(store: CacheStore, metrics: CounterCacheMetrics) -> ResponseCache:
    return ResponseCache(
        store=store,
        generations=CorpusGenerations(store),
        visibility=FakeVisibility(),
        versions=VERSIONS,
        metrics=metrics,
    )


async def test_store_down_response_still_computed() -> None:
    metrics = CounterCacheMetrics()
    cache = response_cache(FailingStore(), metrics)
    work = Counter(answer("d1"))
    got, was_hit = await cache.get_or_compute(scope(), "q", work, authz_version="a")
    again, _ = await cache.get_or_compute(scope(), "q", work, authz_version="a")
    assert got.answer and again.answer and not was_hit and work.calls == 2
    assert metrics.get("response", "error") == 2


async def test_store_down_retrieval_still_computed() -> None:
    store = FailingStore()
    cache = RetrievalCache(
        store=store,
        generations=CorpusGenerations(store),
        visibility=FakeVisibility(),
        config_version="rc1",
    )
    calls = 0

    async def search() -> list[RetrievedChunk]:
        nonlocal calls
        calls += 1
        return [hit("d1")]

    hits, was_hit = await cache.get_or_compute(scope(), "q", 5, search)
    assert hits and not was_hit and calls == 1


async def test_unreachable_redis_degrades_to_compute() -> None:
    client = Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2, socket_timeout=0.2)
    try:
        store = RedisCacheStore(client)
        metrics = CounterCacheMetrics()
        cache = response_cache(store, metrics)
        work = Counter(answer("d1"))
        got, was_hit = await cache.get_or_compute(scope(), "q", work, authz_version="a")
        assert got.answer and not was_hit and work.calls == 1
        assert store.errors >= 1 and metrics.get("response", "error") == 1
        inv = CacheInvalidator(store, CorpusGenerations(store))
        import pytest
        from ekip_caching import CacheInvalidationError

        with pytest.raises(CacheInvalidationError):
            await inv.document_deleted("acme", "d1")
    finally:
        await client.aclose()


async def test_store_error_message_carries_no_connection_details() -> None:
    client = Redis.from_url("redis://user:secretpw@127.0.0.1:1/0", socket_connect_timeout=0.2)
    try:
        import pytest
        from ekip_caching import CacheStoreError

        with pytest.raises(CacheStoreError) as info:
            await RedisCacheStore(client).get("k")
        assert "secretpw" not in str(info.value) and "127.0.0.1" not in str(info.value)
    finally:
        await client.aclose()
