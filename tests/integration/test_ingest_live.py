"""End-to-end ingestion against live Postgres + OpenSearch (+ Gemini when RUN_LIVE_LLM=1)."""

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
from ekip_config.settings import Settings
from ekip_contracts.documents import Document, DocumentAcl, DocumentStatus
from ekip_ingestion import IngestDocument, default_registry
from ekip_llm_gateway import GeminiEmbedder, HashingEmbedder
from ekip_persistence import InMemoryObjectStore, PostgresDocumentRepository, create_pool
from ekip_search import OpenSearchChunkIndex, tenant_and_acl_filter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs live Compose stack"),
]

HTML = (
    b"<html><head><title>Gear</title></head><body><h1>Landing gear</h1>"
    b"<p>The hydraulic pump must be inspected every 400 flight hours.</p>"
    b"<h1>Tyres</h1><p>Tyre pressure is checked before each flight.</p></body></html>"
)


@pytest.fixture
async def rig() -> AsyncIterator[
    tuple[IngestDocument, OpenSearchChunkIndex, PostgresDocumentRepository]
]:
    s = Settings()
    pool = await create_pool(str(s.database_url))
    repo = PostgresDocumentRepository(pool)
    live = os.environ.get("RUN_LIVE_LLM") == "1" and s.google_api_key is not None
    if live:
        assert s.google_api_key is not None
        embedder = GeminiEmbedder(
            s.google_api_key.get_secret_value(),
            model=s.embedding_model,
            dimensions=s.embedding_dimensions,
        )
        dims = embedder.dimensions
    else:
        embedder = HashingEmbedder(32)  # type: ignore[assignment]
        dims = 32
    # a throwaway index so this run never touches (or depends on) the real one
    index = OpenSearchChunkIndex(
        str(s.opensearch_url),
        model_version=embedder.model_version,
        dimensions=dims,
        index_name=f"ekip-it-{uuid.uuid4().hex[:8]}",
    )
    await index.ensure_ready()
    use_case = IngestDocument(
        documents=repo, assets=repo, objects=InMemoryObjectStore(),
        parsers=default_registry(), embedder=embedder, index=index,
    )  # fmt: skip
    try:
        yield use_case, index, repo
    finally:
        async with httpx.AsyncClient(trust_env=True) as c:
            await c.delete(f"{str(s.opensearch_url).rstrip('/')}/{index.index_name}")
        await index.aclose()
        await pool.close()


async def test_ingest_end_to_end_then_search_and_recover(rig) -> None:  # type: ignore[no-untyped-def]
    use_case, index, repo = rig
    tenant = f"it-{uuid.uuid4().hex[:8]}"
    doc = Document.create(
        tenant_id=tenant, source="sharepoint", source_document_id="manual-1", version=1,
        source_version="v1", content=HTML, title="Gear manual", mime_type="text/html",
        source_uri="https://example.test/manual-1",
        acl=DocumentAcl("acl-1", frozenset({"alice"}), frozenset()),
        last_modified=datetime(2026, 10, 19, tzinfo=UTC),
    )  # fmt: skip
    result = await use_case.ingest(doc, HTML)
    assert result.document.status is DocumentStatus.INDEXED
    assert await index.count_chunks(tenant, doc.document_id) == result.chunk_count >= 2
    stored = await repo.get(tenant, doc.document_id, 1)
    assert stored is not None and stored.status is DocumentStatus.INDEXED

    url = f"{str(Settings(_env_file=None).opensearch_url).rstrip('/')}/{index.index_name}/_search"
    query = {
        "query": {
            "bool": {
                "must": {"match": {"content": "hydraulic"}},
                "filter": tenant_and_acl_filter(tenant, "alice", set()),
            }
        }
    }
    async with httpx.AsyncClient(trust_env=True) as c:
        hits = (await c.post(url, json=query)).json()["hits"]["hits"]
        assert len(hits) == 1
        query["query"]["bool"]["filter"] = tenant_and_acl_filter(tenant, "mallory", set())
        assert (await c.post(url, json=query)).json()["hits"]["hits"] == []

    again = await use_case.run(tenant, doc.document_id, 1)  # idempotent
    assert again.chunk_count == result.chunk_count
