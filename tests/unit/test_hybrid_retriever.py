"""HybridRetriever: fusion of both searches, graceful degradation, and what it must never do."""

from collections.abc import Sequence
from typing import Any

import pytest
from ekip_contracts.embedding import EmbeddingTask, EmbeddingUnavailableError
from ekip_contracts.retrieval import RerankError, RetrievedChunk, SearchScope
from ekip_contracts.search_index import SearchIndexUnavailableError
from ekip_llm_gateway import CachingEmbedder, HashingEmbedder, InMemoryEmbeddingCache
from ekip_retrieval import (
    HybridConfig,
    HybridRetriever,
    InvalidQueryError,
    PassThroughReranker,
    SearchMode,
    TermCoverageReranker,
)
from ekip_search import InMemoryChunkIndex
from index_factory import DIMS, MODEL, acl, document, entry, unit

ALICE = SearchScope("tenant-a", "alice")


class SpySearcher:
    """Wraps the in-memory index; records calls and can be told to fail."""

    def __init__(self, index: InMemoryChunkIndex) -> None:
        self.index = index
        self.lexical_calls: list[tuple[str, SearchScope, int]] = []
        self.vector_calls: list[tuple[SearchScope, int]] = []
        self.lexical_error: Exception | None = None
        self.vector_error: Exception | None = None

    async def lexical_search(
        self, query: str, *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        self.lexical_calls.append((query, scope, limit))
        if self.lexical_error:
            raise self.lexical_error
        return await self.index.lexical_search(query, scope=scope, limit=limit)

    async def vector_search(
        self, vector: Sequence[float], *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        self.vector_calls.append((scope, limit))
        if self.vector_error:
            raise self.vector_error
        return await self.index.vector_search(vector, scope=scope, limit=limit)


class FixedQueryEmbedder:
    """Maps every query to one chosen unit vector, so tests control the 'semantic' side."""

    model_version = MODEL
    dimensions = DIMS

    def __init__(self, axis: int) -> None:
        self.axis = axis
        self.calls = 0
        self.error: Exception | None = None

    async def embed(self, texts: Sequence[str], *, task: EmbeddingTask) -> list[tuple[float, ...]]:
        self.calls += 1
        if self.error:
            raise self.error
        assert task is EmbeddingTask.QUERY
        return [tuple(unit(self.axis)) for _ in texts]


class Rig:
    def __init__(self, **cfg: Any) -> None:
        self.index = InMemoryChunkIndex(model_version=MODEL, dimensions=DIMS)
        self.searcher = SpySearcher(self.index)
        self.embedder = FixedQueryEmbedder(axis=2)
        self.cfg = cfg

    def retriever(self, **kw: Any) -> HybridRetriever:
        config = HybridConfig(**self.cfg) if self.cfg else HybridConfig()
        return HybridRetriever(self.searcher, self.embedder, config=config, **kw)

    async def seed(self) -> list[str]:
        """chunk 0 matches the query LEXICALLY; chunk 2 matches it SEMANTICALLY (axis 2)."""
        doc = document(doc_acl=acl(("alice",), ()))
        texts = [
            "torque procedure AMM-32-11 for the pump",
            "cabin lighting",
            "gear retraction cycle",
        ]
        entries = [entry(doc, i, t, unit(i)) for i, t in enumerate(texts)]
        await self.index.replace_document(doc, entries)
        return [e.chunk.chunk_id for e in entries]


QUERY = "AMM-32-11 torque"


async def test_hybrid_surfaces_both_the_lexical_and_the_semantic_match() -> None:
    rig = Rig()
    ids = await rig.seed()
    result = await rig.retriever().search_detailed(QUERY, scope=ALICE, limit=3)
    got = [h.chunk_id for h in result.hits]
    assert got[0] == ids[0] or got[0] == ids[2]
    assert {ids[0], ids[2]} <= set(got[:2])  # the exact-id hit AND the semantic hit lead
    assert (result.trace.lexical_hits, result.trace.vector_hits) == (1, 3)
    assert result.trace.degraded == () and not result.trace.reranked


async def test_modes_run_only_the_searches_they_need() -> None:
    rig = Rig(mode=SearchMode.LEXICAL)
    ids = await rig.seed()
    hits = await rig.retriever().search(QUERY, scope=ALICE)
    assert [h.chunk_id for h in hits] == [ids[0]] and rig.embedder.calls == 0
    assert rig.searcher.vector_calls == []

    rig = Rig(mode=SearchMode.VECTOR)
    ids = await rig.seed()
    hits = await rig.retriever().search(QUERY, scope=ALICE)
    assert hits[0].chunk_id == ids[2] and rig.searcher.lexical_calls == []


async def test_the_callers_scope_reaches_both_searches_unchanged() -> None:
    rig = Rig()
    await rig.seed()
    scope = SearchScope("tenant-a", "alice", frozenset({"eng"}))
    await rig.retriever().search(QUERY, scope=scope)
    assert rig.searcher.lexical_calls[0][1] is scope
    assert rig.searcher.vector_calls[0][0] is scope


async def test_a_caller_no_acl_can_name_costs_no_searches_at_all() -> None:
    rig = Rig()
    await rig.seed()
    result = await rig.retriever().search_detailed(QUERY, scope=SearchScope("tenant-a"))
    assert result.hits == []
    assert rig.searcher.lexical_calls == [] and rig.embedder.calls == 0


async def test_unauthorised_users_and_other_tenants_get_nothing() -> None:
    rig = Rig()
    await rig.seed()
    r = rig.retriever()
    assert await r.search(QUERY, scope=SearchScope("tenant-a", "mallory")) == []
    assert await r.search(QUERY, scope=SearchScope("tenant-b", "alice")) == []


async def test_the_query_is_normalised_before_searching() -> None:
    rig = Rig()
    await rig.seed()
    await rig.retriever().search("  ＡＭＭ-32-11 \n torque ", scope=ALICE)
    assert rig.searcher.lexical_calls[0][0] == "AMM-32-11 torque"


@pytest.mark.parametrize("limit", [0, -1, 51])
async def test_limits_are_validated(limit: int) -> None:
    rig = Rig()
    await rig.seed()
    with pytest.raises(InvalidQueryError):
        await rig.retriever().search(QUERY, scope=ALICE, limit=limit)


async def test_blank_queries_are_rejected_before_any_search() -> None:
    rig = Rig()
    with pytest.raises(InvalidQueryError):
        await rig.retriever().search("  \x00 ", scope=ALICE)
    assert rig.searcher.lexical_calls == [] and rig.embedder.calls == 0


# ---- degradation -----------------------------------------------------------------------------


async def test_a_lexical_outage_falls_back_to_vector_results() -> None:
    rig = Rig()
    ids = await rig.seed()
    rig.searcher.lexical_error = SearchIndexUnavailableError("down")
    result = await rig.retriever().search_detailed(QUERY, scope=ALICE)
    assert result.hits[0].chunk_id == ids[2] and result.trace.degraded == ("lexical_failed",)


async def test_an_embedding_outage_falls_back_to_lexical_results() -> None:
    rig = Rig()
    ids = await rig.seed()
    rig.embedder.error = EmbeddingUnavailableError("rate limited")
    result = await rig.retriever().search_detailed(QUERY, scope=ALICE)
    assert [h.chunk_id for h in result.hits] == [ids[0]]
    assert result.trace.degraded == ("vector_failed",)


async def test_losing_every_needed_search_is_an_error_not_an_empty_answer() -> None:
    rig = Rig()
    await rig.seed()
    rig.searcher.lexical_error = SearchIndexUnavailableError("down")
    rig.embedder.error = EmbeddingUnavailableError("down")
    with pytest.raises(SearchIndexUnavailableError):
        await rig.retriever().search(QUERY, scope=ALICE)
    lexical_only = Rig(mode=SearchMode.LEXICAL)
    await lexical_only.seed()
    lexical_only.searcher.lexical_error = SearchIndexUnavailableError("down")
    with pytest.raises(SearchIndexUnavailableError):
        await lexical_only.retriever().search(QUERY, scope=ALICE)


async def test_bugs_are_never_swallowed_as_degradation() -> None:
    rig = Rig()
    await rig.seed()
    rig.searcher.vector_error = RuntimeError("programming error")
    with pytest.raises(RuntimeError):
        await rig.retriever().search(QUERY, scope=ALICE)


# ---- reranking -------------------------------------------------------------------------------


async def test_a_reranker_reorders_the_candidates() -> None:
    rig = Rig()
    ids = await rig.seed()
    plain = await rig.retriever().search("cabin lighting AMM-32-11 torque", scope=ALICE, limit=3)

    class Reverse:
        async def rerank(
            self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
        ) -> list[RetrievedChunk]:
            return list(reversed(candidates))[:limit]

    result = await rig.retriever(reranker=Reverse()).search_detailed(
        "cabin lighting AMM-32-11 torque", scope=ALICE, limit=3
    )
    assert [h.chunk_id for h in result.hits] == [h.chunk_id for h in reversed(plain)]
    assert result.trace.reranked and set(h.chunk_id for h in result.hits) <= set(ids)


async def test_a_failing_reranker_keeps_the_fused_order() -> None:
    rig = Rig()
    await rig.seed()
    baseline = await rig.retriever().search(QUERY, scope=ALICE)

    class Broken:
        async def rerank(
            self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
        ) -> list[RetrievedChunk]:
            raise RerankError("503")

    result = await rig.retriever(reranker=Broken()).search_detailed(QUERY, scope=ALICE)
    assert [h.chunk_id for h in result.hits] == [h.chunk_id for h in baseline]
    assert "rerank_failed" in result.trace.degraded and not result.trace.reranked


async def test_a_reranker_cannot_add_or_repeat_chunks() -> None:
    rig = Rig()
    ids = await rig.seed()

    class Rogue:
        async def rerank(
            self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
        ) -> list[RetrievedChunk]:
            planted = RetrievedChunk(chunk_id="forbidden", document_id="x", text="leak", score=9)
            return [planted, candidates[0], candidates[0]]

    hits = await rig.retriever(reranker=Rogue()).search(QUERY, scope=ALICE)
    assert [h.chunk_id for h in hits] == [
        (await rig.retriever().search(QUERY, scope=ALICE))[0].chunk_id
    ]
    assert "forbidden" not in {h.chunk_id for h in hits} and ids


async def test_only_the_top_candidates_go_to_the_reranker() -> None:
    rig = Rig(rerank_top_n=2)
    await rig.seed()
    seen: list[int] = []

    class Spy:
        async def rerank(
            self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
        ) -> list[RetrievedChunk]:
            seen.append(len(candidates))
            return list(candidates)

    await rig.retriever(reranker=Spy()).search(QUERY, scope=ALICE, limit=3)
    assert seen == [2]


async def test_baseline_rerankers() -> None:
    a = RetrievedChunk(chunk_id="a", document_id="d", text="pump pump pump", score=1)
    b = RetrievedChunk(chunk_id="b", document_id="d", text="pump torque value", score=1)
    out = await TermCoverageReranker().rerank("pump torque", [a, b], limit=2)
    assert [h.chunk_id for h in out] == ["b", "a"] and out[0].score == 1.0
    assert [h.chunk_id for h in await PassThroughReranker().rerank("x", [a, b], limit=1)] == ["a"]
    tie = await TermCoverageReranker().rerank("pump", [a, b], limit=2)
    assert [h.chunk_id for h in tie] == ["a", "b"]  # equal coverage keeps the fused order


# ---- trace and caching -----------------------------------------------------------------------


async def test_the_trace_has_timings_and_never_the_query() -> None:
    rig = Rig()
    await rig.seed()
    result = await rig.retriever().search_detailed(QUERY, scope=ALICE)
    assert {"lexical_ms", "embed_ms", "vector_ms", "fuse_ms", "total_ms"} <= set(
        result.trace.timings_ms
    )
    assert "AMM" not in repr(result.trace)


async def test_query_embeddings_are_cached_per_tenant() -> None:
    base = HashingEmbedder(DIMS)
    embedder = CachingEmbedder(base, InMemoryEmbeddingCache())
    index = InMemoryChunkIndex(model_version=base.model_version, dimensions=DIMS)
    searcher = SpySearcher(index)
    r = HybridRetriever(searcher, embedder)
    for tenant in ("tenant-a", "tenant-a", "tenant-b"):
        await r.search("torque procedure", scope=SearchScope(tenant, "alice"))
    assert base.texts_embedded == 2  # tenant-a twice -> one embedding; tenant-b embeds its own


def test_config_is_validated() -> None:
    for bad in ({"candidates": 0}, {"rrf_k": 0}, {"lexical_weight": 0.0}, {"max_limit": 99}):
        with pytest.raises(ValueError):
            HybridConfig(**bad)
