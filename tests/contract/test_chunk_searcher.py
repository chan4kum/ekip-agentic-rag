"""One behaviour spec for the search side of the chunk index, run against every implementation.

The in-memory twin always runs; OpenSearch runs with RUN_INTEGRATION=1. The properties that matter
most are security ones: tenant isolation, ACL filtering INSIDE the query, instant revocation.
"""

import os
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Protocol

import httpx
import pytest
from ekip_config.settings import Settings
from ekip_contracts.documents import ContractError, Document, DocumentAcl
from ekip_contracts.retrieval import ChunkSearcher, SearchScope
from ekip_contracts.search_index import ChunkIndex, IndexEntry
from ekip_search import InMemoryChunkIndex, OpenSearchChunkIndex
from index_factory import DIMS, acl, document, entry, unit

LIVE = os.environ.get("RUN_INTEGRATION") == "1"


class Index(ChunkIndex, ChunkSearcher, Protocol):
    """Both ports, as the real adapters provide them."""


@pytest.fixture(params=["memory", pytest.param("opensearch", marks=pytest.mark.integration)])
async def index(request: pytest.FixtureRequest) -> AsyncIterator[Index]:
    model = f"cs-{uuid.uuid4().hex[:8]}@{DIMS}"
    if request.param == "memory":
        yield InMemoryChunkIndex(model_version=model, dimensions=DIMS)
        return
    if not LIVE:
        pytest.skip("needs live Compose stack (RUN_INTEGRATION=1)")
    url = str(Settings(_env_file=None).opensearch_url)
    live = OpenSearchChunkIndex(url, model_version=model, dimensions=DIMS)
    await live.ensure_ready()
    try:
        yield live
    finally:
        async with httpx.AsyncClient(trust_env=True) as c:
            await c.delete(f"{url.rstrip('/')}/{live.index_name}")
        await live.aclose()


def scope(tenant: str = "tenant-a", user: str = "alice", *groups: str) -> SearchScope:
    return SearchScope(tenant, user, frozenset(groups))


async def publish(
    index: Index, doc: Document, texts: Sequence[str], vectors: Sequence[Sequence[float]] = ()
) -> list[IndexEntry]:
    entries = [
        entry(doc, i, t, vectors[i] if vectors else unit(i), model=index.model_version)
        for i, t in enumerate(texts)
    ]
    await index.replace_document(doc, entries)
    return entries


def ids(hits: Sequence[object]) -> list[str]:
    return [h.chunk_id for h in hits]  # type: ignore[attr-defined]


# ---- lexical ---------------------------------------------------------------------------------


async def test_exact_identifiers_match_lexically(index: Index) -> None:
    doc = document()
    entries = await publish(
        index, doc, ["hydraulic pump overhaul AMM-32-11", "tyre pressure check"]
    )
    hits = await index.lexical_search("AMM-32-11", scope=scope(), limit=5)
    assert ids(hits)[0] == entries[0].chunk.chunk_id and hits[0].score > 0


async def test_hits_carry_the_metadata_a_citation_needs(index: Index) -> None:
    doc = document()
    [e] = await publish(index, doc, ["hydraulic pump overhaul"])
    [hit] = await index.lexical_search("hydraulic", scope=scope(), limit=5)
    assert hit.chunk_id == e.chunk.chunk_id and hit.document_id == doc.document_id
    assert hit.text == "hydraulic pump overhaul" and hit.document_version == 1
    assert hit.title == "Landing gear manual" and hit.source_uri == doc.source_uri
    assert hit.section_path == ("Ch 3", "Gear") and hit.content_hash == e.chunk.content_hash


async def test_blank_queries_return_nothing(index: Index) -> None:
    await publish(index, document(), ["hydraulic pump"])
    assert await index.lexical_search("   ", scope=scope(), limit=5) == []
    assert await index.lexical_search("!!!", scope=scope(), limit=5) == []


async def test_limit_is_respected_and_validated(index: Index) -> None:
    await publish(index, document(), [f"pump number {i}" for i in range(6)])
    assert len(await index.lexical_search("pump", scope=scope(), limit=3)) == 3
    for bad in (0, -1, 10_000):
        with pytest.raises(ContractError):
            await index.lexical_search("pump", scope=scope(), limit=bad)


# ---- tenant isolation and ACLs ---------------------------------------------------------------


async def test_other_tenants_documents_are_never_returned(index: Index) -> None:
    await publish(index, document(tenant="tenant-a"), ["secret hydraulic design"])
    await publish(
        index, document(tenant="tenant-b", doc_acl=acl(("alice",), ())), ["secret hydraulic"]
    )
    mine = await index.lexical_search("hydraulic", scope=scope("tenant-a"), limit=10)
    assert len(mine) == 1 and mine[0].document_id == document(tenant="tenant-a").document_id
    vec = await index.vector_search(unit(0), scope=scope("tenant-a"), limit=10)
    assert {h.document_id for h in vec} == {mine[0].document_id}


async def test_user_and_group_grants_both_work_and_strangers_see_nothing(index: Index) -> None:
    await publish(index, document(doc_acl=acl(("alice",), ("eng",))), ["landing gear hydraulic"])
    assert await index.lexical_search("hydraulic", scope=scope(user="alice"), limit=5)
    assert await index.lexical_search("hydraulic", scope=scope("tenant-a", "carol", "eng"), limit=5)
    assert (
        await index.lexical_search("hydraulic", scope=scope("tenant-a", "carol", "sales"), limit=5)
        == []
    )
    assert await index.lexical_search("hydraulic", scope=scope(user="mallory"), limit=5) == []


async def test_a_caller_without_identities_sees_nothing(index: Index) -> None:
    await publish(index, document(doc_acl=acl(("alice",), ("eng",))), ["landing gear hydraulic"])
    nobody = SearchScope("tenant-a")
    assert await index.lexical_search("hydraulic", scope=nobody, limit=5) == []
    assert await index.vector_search(unit(0), scope=nobody, limit=5) == []


async def test_an_empty_acl_matches_nobody(index: Index) -> None:
    await publish(index, document(doc_acl=DocumentAcl("acl-0")), ["landing gear hydraulic"])
    for who in (scope(), scope("tenant-a", "alice", "eng"), scope("tenant-a", "", "eng")):
        assert await index.lexical_search("hydraulic", scope=who, limit=5) == []
        assert await index.vector_search(unit(0), scope=who, limit=5) == []


async def test_revoking_access_takes_effect_immediately_for_both_searches(index: Index) -> None:
    doc = document(doc_acl=acl(("alice",), ()))
    await publish(index, doc, ["landing gear hydraulic"])
    assert await index.lexical_search("hydraulic", scope=scope(), limit=5)
    await index.update_acl("tenant-a", doc.document_id, acl(("bob",), ()))
    assert await index.lexical_search("hydraulic", scope=scope(), limit=5) == []
    assert await index.vector_search(unit(0), scope=scope(), limit=5) == []
    assert await index.lexical_search("hydraulic", scope=scope(user="bob"), limit=5)


# ---- vector ----------------------------------------------------------------------------------


async def test_nearest_vector_ranks_first(index: Index) -> None:
    entries = await publish(index, document(), ["alpha", "bravo", "charlie"])
    hits = await index.vector_search(unit(1), scope=scope(), limit=3)
    assert ids(hits)[0] == entries[1].chunk.chunk_id
    assert hits[0].score >= hits[1].score >= hits[2].score


async def test_acl_is_applied_during_vector_search_not_after(index: Index) -> None:
    """The restricted chunk is the NEAREST one. A post-filter would return fewer than `limit`."""
    restricted = document(source_id="secret", doc_acl=acl(("bob",), ()))
    open_doc = document(source_id="open", doc_acl=acl(("alice",), ()))
    await publish(index, restricted, ["restricted"], [unit(0)])
    entries = await publish(
        index, open_doc, ["open one", "open two", "open three"], [unit(1), unit(2), unit(3)]
    )
    hits = await index.vector_search(unit(0), scope=scope(), limit=3)
    assert sorted(ids(hits)) == sorted(e.chunk.chunk_id for e in entries)


async def test_bad_query_vectors_are_rejected(index: Index) -> None:
    await publish(index, document(), ["alpha"])
    with pytest.raises(ContractError):
        await index.vector_search([1.0, 2.0], scope=scope(), limit=3)
    with pytest.raises(ContractError):
        await index.vector_search([float("nan")] * DIMS, scope=scope(), limit=3)


# ---- lifecycle -------------------------------------------------------------------------------


async def test_deleted_and_replaced_content_is_not_searchable(index: Index) -> None:
    v1 = document(version=1)
    await publish(index, v1, ["old hydraulic procedure"])
    v2 = document(version=2)
    await publish(index, v2, ["new pneumatic procedure"])
    assert await index.lexical_search("hydraulic", scope=scope(), limit=5) == []
    assert len(await index.lexical_search("pneumatic", scope=scope(), limit=5)) == 1
    await index.delete_document("tenant-a", v2.document_id)
    assert await index.lexical_search("pneumatic", scope=scope(), limit=5) == []
    assert await index.vector_search(unit(0), scope=scope(), limit=5) == []
