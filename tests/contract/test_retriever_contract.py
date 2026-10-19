"""Contract tests: EVERY Retriever adapter must pass this suite (Liskov, ADR-006 #5).

The suite is about behaviour a caller can rely on, whatever sits behind the port: tenant isolation,
document ACLs, limits, ordering and the shape of what comes back.
"""

import index_factory as f
import pytest
from ekip_contracts.embedding import EmbeddingTask
from ekip_contracts.retrieval import RetrievedChunk, Retriever, SearchScope
from ekip_llm_gateway.hashing import HashingEmbedder
from ekip_retrieval import HybridRetriever, InvalidQueryError, TermCoverageReranker
from ekip_search.memory import InMemoryChunkIndex
from index_factory import acl, document

ENG = SearchScope("tenant-a", user="alice", groups=frozenset({"eng"}))
OPS = SearchScope("tenant-a", user="oscar", groups=frozenset({"ops"}))
OTHER_TENANT = SearchScope("tenant-b", user="alice", groups=frozenset({"eng"}))
NOBODY = SearchScope("tenant-a")

DOCS = [
    ("tenant-a", "d1", ("eng",), [
        "torque value for fastener inspection",
        "inspection interval for landing gear",
    ]),
    ("tenant-a", "d2", ("eng",), ["cabin lighting procedure"]),
    ("tenant-a", "d3", ("ops",), ["restricted torque inspection audit"]),
    ("tenant-b", "d4", ("eng",), ["secret torque inspection document of tenant b"]),
]  # fmt: skip


class RetrieverContract:
    async def make_retriever(self) -> Retriever:
        raise NotImplementedError

    @pytest.fixture
    async def retriever(self) -> Retriever:
        return await self.make_retriever()

    async def test_returns_only_the_callers_tenant(self, retriever: Retriever) -> None:
        results = await retriever.search("torque inspection", scope=ENG)
        assert results and {r.document_id for r in results} <= {
            _doc("tenant-a", "d1"),
            _doc("tenant-a", "d2"),
        }
        assert all(r.document_id != _doc("tenant-b", "d4") for r in results)

    async def test_document_acl_is_enforced_inside_a_tenant(self, retriever: Retriever) -> None:
        eng = {r.document_id for r in await retriever.search("torque inspection", scope=ENG)}
        ops = {r.document_id for r in await retriever.search("torque inspection", scope=OPS)}
        assert _doc("tenant-a", "d3") not in eng and _doc("tenant-a", "d3") in ops
        assert _doc("tenant-a", "d1") not in ops

    async def test_a_caller_with_no_identity_gets_nothing(self, retriever: Retriever) -> None:
        assert await retriever.search("torque", scope=NOBODY) == []

    async def test_unknown_tenant_gets_nothing(self, retriever: Retriever) -> None:
        scope = SearchScope("nobody", user="alice", groups=frozenset({"eng"}))
        assert await retriever.search("torque", scope=scope) == []

    async def test_respects_limit(self, retriever: Retriever) -> None:
        assert len(await retriever.search("inspection", scope=ENG, limit=1)) == 1

    async def test_sorted_by_score_descending(self, retriever: Retriever) -> None:
        scores = [r.score for r in await retriever.search("torque inspection", scope=ENG)]
        assert scores == sorted(scores, reverse=True)

    async def test_no_repeated_chunks(self, retriever: Retriever) -> None:
        ids = [r.chunk_id for r in await retriever.search("torque inspection landing", scope=ENG)]
        assert len(ids) == len(set(ids))

    async def test_returns_citable_retrieved_chunks(self, retriever: Retriever) -> None:
        results = await retriever.search("torque", scope=ENG)
        assert results and all(isinstance(r, RetrievedChunk) for r in results)
        assert all(r.document_version == 1 and r.title and r.source_uri for r in results)

    async def test_blank_queries_are_refused(self, retriever: Retriever) -> None:
        with pytest.raises(InvalidQueryError):
            await retriever.search("   ", scope=ENG)


def _doc(tenant: str, source_id: str) -> str:
    return document(tenant=tenant, source_id=source_id).document_id


async def _hybrid(*, rerank: bool) -> Retriever:
    embedder = HashingEmbedder(f.DIMS)
    index = InMemoryChunkIndex(model_version=embedder.model_version, dimensions=f.DIMS)
    for tenant, source_id, groups, texts in DOCS:
        doc = document(tenant=tenant, source_id=source_id, doc_acl=acl(users=(), groups=groups))
        vectors = await embedder.embed(texts, task=EmbeddingTask.DOCUMENT)
        await index.replace_document(
            doc,
            [
                f.entry(doc, i, t, v, model=embedder.model_version)
                for i, (t, v) in enumerate(zip(texts, vectors, strict=True))
            ],
        )
    return HybridRetriever(index, embedder, reranker=TermCoverageReranker() if rerank else None)


class TestHybridWithoutReranker(RetrieverContract):
    async def make_retriever(self) -> Retriever:
        return await _hybrid(rerank=False)


class TestHybridWithReranker(RetrieverContract):
    async def make_retriever(self) -> Retriever:
        return await _hybrid(rerank=True)
