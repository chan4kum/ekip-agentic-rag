"""Contract tests: EVERY Retriever adapter must pass this suite (Liskov, ADR-006 #5)."""

from collections.abc import Sequence

import pytest
from ekip_api.adapters.in_memory_retriever import InMemoryRetriever, StoredChunk
from ekip_contracts.retrieval import RetrievedChunk, Retriever

CHUNKS = [
    StoredChunk("tenant-a", "c1", "d1", "torque value for fastener inspection"),
    StoredChunk("tenant-a", "c2", "d1", "inspection interval for landing gear"),
    StoredChunk("tenant-a", "c3", "d2", "cabin lighting procedure"),
    StoredChunk("tenant-b", "c4", "d3", "secret torque inspection document of tenant b"),
]


class RetrieverContract:
    def make_retriever(self, chunks: Sequence[StoredChunk]) -> Retriever:
        raise NotImplementedError

    @pytest.fixture
    def retriever(self) -> Retriever:
        return self.make_retriever(CHUNKS)

    async def test_returns_only_the_callers_tenant(self, retriever: Retriever) -> None:
        results = await retriever.search("torque inspection", tenant_id="tenant-a")
        assert {r.chunk_id for r in results} == {"c1", "c2"}

    async def test_unknown_tenant_gets_nothing(self, retriever: Retriever) -> None:
        assert await retriever.search("torque", tenant_id="nobody") == []

    async def test_respects_limit(self, retriever: Retriever) -> None:
        results = await retriever.search("inspection", tenant_id="tenant-a", limit=1)
        assert len(results) == 1

    async def test_sorted_by_score_descending(self, retriever: Retriever) -> None:
        results = await retriever.search("torque inspection", tenant_id="tenant-a")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].chunk_id == "c1"

    async def test_no_match_returns_empty_list(self, retriever: Retriever) -> None:
        assert await retriever.search("zebra", tenant_id="tenant-a") == []

    async def test_returns_retrieved_chunks(self, retriever: Retriever) -> None:
        results = await retriever.search("torque", tenant_id="tenant-a")
        assert results
        assert all(isinstance(r, RetrievedChunk) for r in results)


class TestInMemoryRetriever(RetrieverContract):
    def make_retriever(self, chunks: Sequence[StoredChunk]) -> Retriever:
        return InMemoryRetriever(chunks)
