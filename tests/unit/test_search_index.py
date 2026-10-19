import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from ekip_contracts.documents import ContractError
from ekip_contracts.search_index import (
    SearchIndexError,
    SearchIndexUnavailableError,
    StaleIndexVersionError,
)
from ekip_search import OpenSearchChunkIndex, index_body, index_name_for, tenant_and_acl_filter
from index_factory import DIMS, MODEL, document, entry, unit

Handler = Callable[[httpx.Request], httpx.Response]


def make_index(handler: Handler, **kw: Any) -> OpenSearchChunkIndex:
    async def no_sleep(_: float) -> None:
        return None

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenSearchChunkIndex(
        "http://os:9200", model_version=MODEL, dimensions=DIMS, client=client,
        sleep=no_sleep, jitter=lambda: 1.0, **kw,
    )  # fmt: skip


class FakeCluster:
    """Just enough OpenSearch to exercise the adapter's request logic."""

    def __init__(self, newest: int | None = None) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.newest = newest
        self.bulk_errors = False
        self.index_exists = False
        self.meta: dict[str, Any] | None = None

    def __call__(self, r: httpx.Request) -> httpx.Response:
        path = r.url.path
        body: Any = None
        if r.content:
            text = r.content.decode()
            body = text if path == "/_bulk" else json.loads(text)
        self.requests.append((r.method, path, body))
        if path.endswith("/_mapping"):
            if not self.index_exists:
                return httpx.Response(404, json={})
            name = index_name_for(MODEL)
            return httpx.Response(200, json={name: {"mappings": {"_meta": self.meta}}})
        if r.method == "PUT":
            self.index_exists = True
            self.meta = body["mappings"]["_meta"]
            return httpx.Response(200, json={"acknowledged": True})
        if path == "/_bulk":
            return httpx.Response(200, json={"errors": self.bulk_errors, "items": []})
        if path.endswith("/_search"):
            value = None if self.newest is None else float(self.newest)
            return httpx.Response(200, json={"aggregations": {"newest": {"value": value}}})
        if path.endswith("/_count"):
            return httpx.Response(200, json={"count": 2})
        return httpx.Response(200, json={"deleted": 0, "updated": 0})

    def paths(self) -> list[str]:
        return [p for _, p, _ in self.requests]


# ---- mapping / filters -----------------------------------------------------------------------


def test_index_name_is_derived_from_the_model_version() -> None:
    assert index_name_for("gemini-embedding-001@768") == "ekip-chunks-gemini-embedding-001-768"
    assert index_name_for("Weird/Model:1@3") != index_name_for("gemini-embedding-001@768")


def test_mapping_is_strict_and_vector_field_matches_the_model() -> None:
    body = index_body(dimensions=768, model_version="m@768")
    props = body["mappings"]["properties"]
    assert body["mappings"]["dynamic"] == "strict"
    assert props["embedding"]["dimension"] == 768
    assert props["embedding"]["method"]["engine"] == "lucene"
    assert props["embedding"]["method"]["space_type"] == "cosinesimil"
    assert body["settings"]["index"]["knn"] is True
    # filters are keywords (exact), lexical fields are text (BM25)
    for f in ("tenant_id", "document_id", "allowed_users", "allowed_groups"):
        assert props[f]["type"] == "keyword"
    for f in ("content", "title", "section_path"):
        assert props[f]["type"] == "text"


def test_acl_filter_requires_tenant_and_user_or_group() -> None:
    f = tenant_and_acl_filter("t1", "alice", {"eng", "ops"})
    assert f[0] == {"term": {"tenant_id": "t1"}}
    should = f[1]["bool"]["should"]
    assert {"term": {"allowed_users": "alice"}} in should
    assert {"terms": {"allowed_groups": ["eng", "ops"]}} in should
    assert f[1]["bool"]["minimum_should_match"] == 1


def test_acl_filter_without_groups_only_matches_the_user() -> None:
    f = tenant_and_acl_filter("t1", "alice", frozenset())
    assert f[1]["bool"]["should"] == [{"term": {"allowed_users": "alice"}}]


# ---- ensure_ready ----------------------------------------------------------------------------


async def test_ensure_ready_creates_a_missing_index_once() -> None:
    cluster = FakeCluster()
    idx = make_index(cluster)
    await idx.ensure_ready()
    await idx.ensure_ready()
    assert [m for m, p, _ in cluster.requests if m == "PUT"] == ["PUT"]


async def test_ensure_ready_refuses_an_incompatible_existing_index() -> None:
    cluster = FakeCluster()
    cluster.index_exists = True
    cluster.meta = {"mapping_version": 1, "embedding_model_version": MODEL, "dimensions": DIMS + 1}
    with pytest.raises(SearchIndexError, match="incompatible"):
        await make_index(cluster).ensure_ready()


# ---- replace_document ------------------------------------------------------------------------


async def test_publication_writes_chunks_routed_by_tenant_then_prunes_old_ones() -> None:
    cluster = FakeCluster(newest=1)
    doc = document(version=2)
    idx = make_index(cluster)
    n = await idx.replace_document(
        doc, [entry(doc, 0, "alpha", unit(0)), entry(doc, 1, "beta", unit(1))]
    )
    assert n == 2
    bulk = next(b for _, p, b in cluster.requests if p == "/_bulk")
    action, source = (json.loads(x) for x in bulk.strip().split("\n")[:2])
    assert action["index"]["routing"] == "tenant-a"
    assert action["index"]["_id"] == source["chunk_id"]
    assert source["allowed_users"] == ["alice"] and source["embedding_model_version"] == MODEL
    order = cluster.paths()
    assert order.index("/_bulk") < order.index(f"/{index_name_for(MODEL)}/_delete_by_query")
    prune = next(b for _, p, b in cluster.requests if p.endswith("_delete_by_query"))
    assert {"term": {"tenant_id": "tenant-a"}} in prune["query"]["bool"]["filter"]


async def test_older_version_never_overwrites_a_newer_one() -> None:
    cluster = FakeCluster(newest=5)
    doc = document(version=3)
    with pytest.raises(StaleIndexVersionError):
        await make_index(cluster).replace_document(doc, [entry(doc, 0, "x", unit(0))])
    assert "/_bulk" not in cluster.paths()


async def test_large_publications_are_sent_in_bulk_batches() -> None:
    cluster = FakeCluster()
    doc = document()
    entries = [entry(doc, i, f"text {i}", unit(i)) for i in range(450)]
    await make_index(cluster).replace_document(doc, entries)
    assert cluster.paths().count("/_bulk") == 3


async def test_bulk_item_errors_fail_the_publication() -> None:
    cluster = FakeCluster()
    cluster.bulk_errors = True
    doc = document()
    with pytest.raises(SearchIndexError):
        await make_index(cluster).replace_document(doc, [entry(doc, 0, "x", unit(0))])


@pytest.mark.parametrize("case", ["model", "dims", "other_doc", "duplicate"])
async def test_invalid_entries_are_rejected_before_any_request(case: str) -> None:
    def never(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    doc, other = document(), document(source_id="doc-2")
    entries = {
        "model": [entry(doc, 0, "x", unit(0), model="other@32")],
        "dims": [entry(doc, 0, "x", unit(0, 16), model=MODEL)],
        "other_doc": [entry(other, 0, "x", unit(0))],
        "duplicate": [entry(doc, 0, "x", unit(0)), entry(doc, 0, "x", unit(0))],
    }[case]
    with pytest.raises(ContractError):
        await make_index(never).replace_document(doc, entries)


# ---- failure behaviour -----------------------------------------------------------------------


async def test_transient_errors_are_retried_then_reported_as_unavailable() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="document text echoed here")

    with pytest.raises(SearchIndexUnavailableError) as exc:
        await make_index(handler, max_attempts=3).count_chunks("tenant-a", "doc_" + "a" * 32)
    assert calls == 3 and "echoed" not in str(exc.value)


async def test_client_errors_are_not_retried_and_hide_the_body() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, text="secret document text")

    with pytest.raises(SearchIndexError) as exc:
        await make_index(handler).count_chunks("tenant-a", "doc_" + "a" * 32)
    assert calls == 1 and "secret" not in str(exc.value)


async def test_update_acl_sends_new_principals_as_script_params() -> None:
    cluster = FakeCluster()
    from ekip_contracts.documents import DocumentAcl

    await make_index(cluster).update_acl(
        "tenant-a", "doc_" + "a" * 32, DocumentAcl("acl-2", frozenset({"bob"}), frozenset())
    )
    _, _, body = next(r for r in cluster.requests if r[1].endswith("_update_by_query"))
    assert body["script"]["params"] == {"users": ["bob"], "groups": [], "version": "acl-2"}
