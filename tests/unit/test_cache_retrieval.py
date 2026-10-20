from cache_testkit import hit, make_stack, scope
from ekip_caching import CachedRetriever
from ekip_contracts.retrieval import RetrievedChunk, SearchScope


class CountingRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, SearchScope, int]] = []

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.calls.append((query, scope, limit))
        return self.results


async def test_hits_are_cached_and_round_trip_intact() -> None:
    s = make_stack()
    inner = CountingRetriever([hit("d1"), hit("d2", "c3")])
    r = CachedRetriever(inner, s.retrieval)
    first = await r.search("torque", scope=scope(), limit=5)
    second = await r.search("TORQUE ", scope=scope(), limit=5)
    assert first == second == inner.results and len(inner.calls) == 1


async def test_limit_tenant_and_identity_are_part_of_the_key() -> None:
    s = make_stack()
    inner = CountingRetriever([hit("d1")])
    r = CachedRetriever(inner, s.retrieval)
    await r.search("q", scope=scope("acme", "alice", "g"), limit=5)
    await r.search("q", scope=scope("acme", "alice", "g"), limit=6)
    await r.search("q", scope=scope("globex", "alice", "g"), limit=5)
    await r.search("q", scope=scope("acme", "alice", "other"), limit=5)
    assert len(inner.calls) == 4


async def test_inner_retriever_receives_the_callers_own_scope() -> None:
    s = make_stack()
    inner = CountingRetriever([hit("d1")])
    me = scope("acme", "alice", "g")
    await CachedRetriever(inner, s.retrieval).search("q", scope=me, limit=3)
    assert inner.calls == [("q", me, 3)]


async def test_empty_results_are_not_cached() -> None:
    s = make_stack()
    inner = CountingRetriever([])
    r = CachedRetriever(inner, s.retrieval)
    await r.search("q", scope=scope())
    await r.search("q", scope=scope())
    assert len(inner.calls) == 2


async def test_revoked_document_makes_cached_hits_stale() -> None:
    s = make_stack()
    inner = CountingRetriever([hit("d1"), hit("d2")])
    r = CachedRetriever(inner, s.retrieval)
    await r.search("q", scope=scope())
    s.visibility.hidden.add("d2")
    await r.search("q", scope=scope())
    assert len(inner.calls) == 2 and s.metrics.get("retrieval", "stale") == 1


async def test_deletion_bumps_generation_for_retrieval_too() -> None:
    s = make_stack()
    inner = CountingRetriever([hit("d1")])
    r = CachedRetriever(inner, s.retrieval)
    await r.search("q", scope=scope())
    await s.invalidator.document_deleted("acme", "d1")
    await r.search("q", scope=scope())
    assert len(inner.calls) == 2


async def test_visibility_outage_recomputes() -> None:
    s = make_stack()
    inner = CountingRetriever([hit("d1")])
    r = CachedRetriever(inner, s.retrieval)
    await r.search("q", scope=scope())
    s.visibility.broken = True
    assert await r.search("q", scope=scope()) == inner.results
    assert len(inner.calls) == 2


async def test_ttl_expiry() -> None:
    s = make_stack(ttl_s=10)
    inner = CountingRetriever([hit("d1")])
    r = CachedRetriever(inner, s.retrieval)
    await r.search("q", scope=scope())
    s.clock.now += 11
    await r.search("q", scope=scope())
    assert len(inner.calls) == 2
