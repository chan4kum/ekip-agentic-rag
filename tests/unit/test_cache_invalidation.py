import pytest
from cache_testkit import Counter, FailingStore, answer, make_stack, scope
from ekip_caching import (
    CacheInvalidationError,
    CacheInvalidator,
    CorpusGenerations,
    MemoryCacheStore,
)

Q = "q"


async def test_publish_purges_only_entries_citing_that_document() -> None:
    s = make_stack()
    on_d1 = Counter(answer("d1"))
    on_d2 = Counter(answer("d2"))
    await s.response.get_or_compute(scope(), "about d1", on_d1, authz_version="a")
    await s.response.get_or_compute(scope(), "about d2", on_d2, authz_version="a")
    await s.invalidator.document_published("acme", "d1")
    _, hit1 = await s.response.get_or_compute(scope(), "about d1", on_d1, authz_version="a")
    _, hit2 = await s.response.get_or_compute(scope(), "about d2", on_d2, authz_version="a")
    assert not hit1 and hit2


async def test_publish_purge_spans_users_and_layers() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope("acme", "a"), Q, work, authz_version="a")
    await s.response.get_or_compute(scope("acme", "b"), Q, work, authz_version="a")
    purged = await s.store.delete_tag("ekip:cache:tag:acme:d1")
    assert purged == 2


async def test_publish_does_not_bump_by_default_but_can() -> None:
    s = make_stack()
    work = Counter(answer("d2"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a")
    await s.invalidator.document_published("acme", "d1")  # unrelated document
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a")
    assert hit
    bumping = CacheInvalidator(s.store, s.generations, bump_on_publish=True)
    await bumping.document_published("acme", "d1")
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a")
    assert not hit


async def test_tenant_reset_retires_everything_for_that_tenant_only() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope("acme"), Q, work, authz_version="a")
    await s.response.get_or_compute(scope("globex"), Q, work, authz_version="a")
    await s.invalidator.tenant_reset("acme")
    assert not (await s.response.get_or_compute(scope("acme"), Q, work, authz_version="a"))[1]
    assert (await s.response.get_or_compute(scope("globex"), Q, work, authz_version="a"))[1]


async def test_generation_is_monotonic_and_never_starts_at_zero() -> None:
    store = MemoryCacheStore()
    gens = CorpusGenerations(store, now_ms=lambda: 1_700_000_000_000)
    first = await gens.current("acme")
    assert first == 1_700_000_000_000 and await gens.current("acme") == first
    assert await gens.bump("acme") == first + 1
    assert await gens.current("acme") == first + 1
    assert await gens.current("globex") == first  # per tenant


async def test_lost_counter_restarts_from_a_later_timestamp_not_an_old_value() -> None:
    store = MemoryCacheStore()
    now = [1_000]
    gens = CorpusGenerations(store, now_ms=lambda: now[0])
    old = await gens.current("acme")
    store._counters.clear()  # noqa: SLF001 - simulate eviction / flush of the counter only
    now[0] = 5_000
    assert await gens.current("acme") > old


async def test_failed_bump_raises_so_the_caller_retries() -> None:
    store = FailingStore()
    inv = CacheInvalidator(store, CorpusGenerations(store))
    with pytest.raises(CacheInvalidationError):
        await inv.document_deleted("acme", "d1")
    with pytest.raises(CacheInvalidationError):
        await inv.acl_changed("acme", "d1")
    with pytest.raises(CacheInvalidationError):
        await inv.document_published("acme", "d1")
    with pytest.raises(CacheInvalidationError):
        await inv.tenant_reset("acme")


async def test_purge_failure_after_a_successful_bump_is_counted_not_raised() -> None:
    s = make_stack()

    async def broken_delete_tag(tag: str) -> int:
        from ekip_caching import CacheStoreError

        raise CacheStoreError("down")

    s.store.delete_tag = broken_delete_tag  # type: ignore[method-assign]
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a")
    await s.invalidator.document_deleted("acme", "d1")
    assert s.invalidator.purge_errors == 1
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a")
    assert not hit  # unreachable because of the bump
