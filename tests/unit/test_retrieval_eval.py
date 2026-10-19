"""The evaluation harness itself, plus the offline regression gate on the versioned dataset."""

from pathlib import Path

import pytest
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_llm_gateway.hashing import HashingEmbedder
from ekip_retrieval import (
    HybridConfig,
    HybridRetriever,
    SearchMode,
    TermCoverageReranker,
    evaluate,
    ingest_dataset,
    load_dataset,
)
from ekip_retrieval.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from ekip_search.memory import InMemoryChunkIndex

DATASET = Path(__file__).parents[2] / "evaluations" / "datasets" / "retrieval_v1.json"
REL = frozenset({"a", "b"})


def test_recall_counts_relevant_passages_found_in_the_top_k() -> None:
    assert recall_at_k(["a", "x", "b"], REL, 3) == 1.0
    assert recall_at_k(["a", "x", "b"], REL, 2) == 0.5
    assert recall_at_k(["x", "y"], REL, 5) == 0.0


def test_mrr_is_the_inverse_rank_of_the_first_relevant_passage() -> None:
    assert reciprocal_rank(["x", "b", "a"], REL, 5) == 0.5
    assert reciprocal_rank(["x", "y", "b"], REL, 2) == 0.0  # outside the cut-off


def test_ndcg_rewards_ranking_relevant_passages_first() -> None:
    perfect, worse = ndcg_at_k(["a", "b", "x"], REL, 3), ndcg_at_k(["x", "a", "b"], REL, 3)
    assert perfect == pytest.approx(1.0) and 0 < worse < perfect
    assert ndcg_at_k(["x", "y", "z"], REL, 3) == 0.0


def test_the_dataset_is_internally_consistent() -> None:
    ds = load_dataset(DATASET)
    assert ds.version == "retrieval-v1" and len(ds.documents) >= 10
    assert {q.kind for q in ds.queries} >= {
        "exact",
        "paraphrase",
        "multi",
        "restricted",
        "cross_tenant",
    }
    assert len({q.id for q in ds.queries}) == len(ds.queries)
    assert any(not q.relevant for q in ds.queries)  # restricted queries exist
    assert len({q.scope.tenant_id for q in ds.queries}) == 2  # and more than one tenant


async def _setup() -> tuple[InMemoryChunkIndex, HashingEmbedder, dict[str, str]]:
    ds = load_dataset(DATASET)
    embedder = HashingEmbedder(64)
    index = InMemoryChunkIndex(model_version=embedder.model_version, dimensions=64)
    return index, embedder, await ingest_dataset(index, embedder, ds)


async def test_no_variant_ever_returns_a_passage_the_caller_may_not_read() -> None:
    ds = load_dataset(DATASET)
    index, embedder, mapping = await _setup()
    for mode in SearchMode:
        for reranker in (None, TermCoverageReranker()):
            retriever = HybridRetriever(
                index, embedder, reranker=reranker, config=HybridConfig(mode=mode)
            )
            report = await evaluate(f"{mode}", retriever, ds, mapping)
            assert report.leaks == () and report.failures == ()
            assert report.restricted_queries >= 4


async def test_hybrid_is_not_worse_than_either_single_mode_on_this_dataset() -> None:
    """A regression gate for the fusion logic on a small synthetic set (not a quality claim)."""
    ds = load_dataset(DATASET)
    index, embedder, mapping = await _setup()
    recall = {}
    for mode in SearchMode:
        r = HybridRetriever(index, embedder, config=HybridConfig(mode=mode))
        recall[mode] = (await evaluate(str(mode), r, ds, mapping)).overall.recall
    assert recall[SearchMode.HYBRID] >= max(recall[SearchMode.LEXICAL], recall[SearchMode.VECTOR])


async def test_the_leak_check_catches_a_retriever_that_ignores_access_control() -> None:
    ds = load_dataset(DATASET)
    index, embedder, mapping = await _setup()
    everyone = {  # a broken "retriever": searches as a caller who can read everything in the tenant
        t: SearchScope(t, user="root", groups=frozenset({"eng", "ops", "hr", "finance"}))
        for t in ("eval-a", "eval-b")
    }
    honest = HybridRetriever(index, embedder)

    class Leaky:
        async def search(
            self, query: str, *, scope: SearchScope, limit: int = 5
        ) -> list[RetrievedChunk]:
            return await honest.search(query, scope=everyone[scope.tenant_id], limit=limit)

    report = await evaluate("leaky", Leaky(), ds, mapping)
    assert report.leaks, "the harness must notice passages returned beyond the caller's rights"
    assert any(entry.startswith("r1 ->") for entry in report.leaks)


async def test_a_failing_query_is_reported_not_skipped() -> None:
    ds = load_dataset(DATASET)
    _, _, mapping = await _setup()

    class Broken:
        async def search(
            self, query: str, *, scope: SearchScope, limit: int = 5
        ) -> list[RetrievedChunk]:
            raise RuntimeError("down")

    report = await evaluate("broken", Broken(), ds, mapping)
    assert len(report.failures) == len(ds.queries) and report.overall.queries == 0
