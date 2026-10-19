"""Live OpenSearch checks. Run with RUN_INTEGRATION=1 (needs the Compose stack)."""

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from ekip_config.settings import Settings
from ekip_contracts.search_index import StaleIndexVersionError
from ekip_search import OpenSearchChunkIndex, tenant_and_acl_filter
from index_factory import DIMS, acl, document, entry, unit

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs live Compose stack"),
]


@pytest.fixture
async def idx() -> AsyncIterator[OpenSearchChunkIndex]:
    url = str(Settings(_env_file=None).opensearch_url)
    model = f"test-{uuid.uuid4().hex[:8]}@{DIMS}"  # its own index per test
    index = OpenSearchChunkIndex(url, model_version=model, dimensions=DIMS)
    await index.ensure_ready()
    try:
        yield index
    finally:
        async with httpx.AsyncClient(trust_env=True) as c:
            await c.delete(f"{url.rstrip('/')}/{index.index_name}")
        await index.aclose()


async def entries_for(doc: Any, idx: OpenSearchChunkIndex, texts: list[str]) -> list[Any]:
    return [entry(doc, i, t, unit(i), model=idx.model_version) for i, t in enumerate(texts)]


async def search(idx: OpenSearchChunkIndex, body: dict[str, Any]) -> list[str]:
    url = str(Settings(_env_file=None).opensearch_url).rstrip("/")
    async with httpx.AsyncClient(trust_env=True) as c:
        r = await c.post(f"{url}/{idx.index_name}/_search", json=body)
    assert r.status_code == 200, r.text
    return [h["_source"]["chunk_id"] for h in r.json()["hits"]["hits"]]


def bm25(text: str, tenant: str, user: str, groups: set[str]) -> dict[str, Any]:
    return {
        "query": {
            "bool": {
                "must": {"multi_match": {"query": text, "fields": ["content", "title^2"]}},
                "filter": tenant_and_acl_filter(tenant, user, groups),
            }
        }
    }


async def test_ensure_ready_is_idempotent(idx: OpenSearchChunkIndex) -> None:
    await idx.ensure_ready()


async def test_publish_then_bm25_search_respects_tenant_and_acl(idx: OpenSearchChunkIndex) -> None:
    doc = document()  # alice or group eng
    n = await idx.replace_document(
        doc, await entries_for(doc, idx, ["hydraulic pump overhaul AMM-32-11", "tyre pressure"])
    )
    assert n == 2
    assert len(await search(idx, bm25("AMM-32-11", "tenant-a", "alice", set()))) == 1  # exact id
    assert len(await search(idx, bm25("hydraulic", "tenant-a", "carol", {"eng"}))) == 1  # by group
    assert await search(idx, bm25("hydraulic", "tenant-a", "carol", {"sales"})) == []  # no access
    assert await search(idx, bm25("hydraulic", "tenant-b", "alice", {"eng"})) == []  # other tenant


async def test_empty_acl_is_visible_to_nobody(idx: OpenSearchChunkIndex) -> None:
    doc = document(doc_acl=acl(users=(), groups=()))
    await idx.replace_document(doc, await entries_for(doc, idx, ["secret plan"]))
    assert await idx.count_chunks("tenant-a", doc.document_id) == 1
    assert await search(idx, bm25("secret", "tenant-a", "alice", {"eng"})) == []


async def test_knn_filter_applies_during_search_not_after(idx: OpenSearchChunkIndex) -> None:
    allowed, hidden = (
        document(source_id="open"),
        document(source_id="hidden", doc_acl=acl(("zed",), ())),
    )
    await idx.replace_document(
        allowed, [entry(allowed, 0, "allowed far", unit(5), model=idx.model_version)]
    )
    # the forbidden documents are the nearest neighbours of the query vector
    await idx.replace_document(
        hidden,
        [entry(hidden, i, f"hidden {i}", unit(0), model=idx.model_version) for i in range(5)],
    )
    q = {
        "size": 3,
        "query": {
            "knn": {
                "embedding": {
                    "vector": unit(0),
                    "k": 3,
                    "filter": {
                        "bool": {"filter": tenant_and_acl_filter("tenant-a", "alice", set())}
                    },
                }
            }
        },
    }
    assert await search(idx, q) == [allowed.document_id + ":v1:c0000"]


async def test_new_version_replaces_old_and_old_cannot_come_back(idx: OpenSearchChunkIndex) -> None:
    v1, v2 = document(version=1), document(version=2)
    await idx.replace_document(v1, await entries_for(v1, idx, ["old one", "old two", "old three"]))
    assert await idx.replace_document(v2, await entries_for(v2, idx, ["new one"])) == 1
    assert await search(idx, bm25("old", "tenant-a", "alice", set())) == []
    with pytest.raises(StaleIndexVersionError):
        await idx.replace_document(v1, await entries_for(v1, idx, ["old one"]))
    assert await idx.count_chunks("tenant-a", v2.document_id) == 1


async def test_republishing_same_version_with_fewer_chunks_drops_leftovers(
    idx: OpenSearchChunkIndex,
) -> None:
    doc = document()
    await idx.replace_document(doc, await entries_for(doc, idx, ["a one", "b two", "c three"]))
    assert await idx.replace_document(doc, await entries_for(doc, idx, ["a one"])) == 1


async def test_acl_change_takes_effect_without_reembedding(idx: OpenSearchChunkIndex) -> None:
    doc = document()
    await idx.replace_document(doc, await entries_for(doc, idx, ["pump manual"]))
    assert await search(idx, bm25("pump", "tenant-a", "alice", set()))
    assert await idx.update_acl("tenant-a", doc.document_id, acl(users=("bob",), groups=())) == 1
    assert await search(idx, bm25("pump", "tenant-a", "alice", {"eng"})) == []
    assert await search(idx, bm25("pump", "tenant-a", "bob", set()))


async def test_delete_removes_every_chunk_and_is_repeatable(idx: OpenSearchChunkIndex) -> None:
    doc = document()
    await idx.replace_document(doc, await entries_for(doc, idx, ["one", "two"]))
    await idx.delete_document("tenant-a", doc.document_id)
    await idx.delete_document("tenant-a", doc.document_id)
    assert await idx.count_chunks("tenant-a", doc.document_id) == 0


async def test_one_tenant_cannot_delete_or_count_anothers_document(
    idx: OpenSearchChunkIndex,
) -> None:
    doc = document()
    await idx.replace_document(doc, await entries_for(doc, idx, ["one"]))
    await idx.delete_document("tenant-b", doc.document_id)
    assert await idx.count_chunks("tenant-b", doc.document_id) == 0
    assert await idx.count_chunks("tenant-a", doc.document_id) == 1


async def test_unknown_fields_are_rejected_by_the_strict_mapping(idx: OpenSearchChunkIndex) -> None:
    url = str(Settings(_env_file=None).opensearch_url).rstrip("/")
    async with httpx.AsyncClient(trust_env=True) as c:
        r = await c.post(f"{url}/{idx.index_name}/_doc", json={"surprise": 1})
    assert r.status_code == 400
