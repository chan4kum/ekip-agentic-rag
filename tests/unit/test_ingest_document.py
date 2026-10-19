"""IngestDocument: the happy path and, mostly, what happens when things go wrong."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest
from doc_factory import data_uri, png
from ekip_contracts.document_store import VersionConflictError
from ekip_contracts.documents import (
    ContractError,
    Document,
    DocumentAcl,
    DocumentStatus,
)
from ekip_contracts.embedding import EmbeddingError, EmbeddingTask, EmbeddingUnavailableError
from ekip_contracts.object_store import ObjectKey, ObjectKind
from ekip_contracts.parsing import ParsedDocument
from ekip_contracts.search_index import SearchIndexError, SearchIndexUnavailableError
from ekip_ingestion import (
    ImageEnrichment,
    IngestDocument,
    IngestionFailed,
    ParserRegistry,
    default_registry,
)
from ekip_llm_gateway import CachingEmbedder, HashingEmbedder, InMemoryEmbeddingCache
from ekip_persistence import InMemoryDocumentRepository, InMemoryObjectStore
from ekip_search import InMemoryChunkIndex

S = DocumentStatus
TENANT = "tenant-a"
ACL = DocumentAcl("acl-1", frozenset({"alice"}), frozenset({"eng"}))
HTML = (
    b"<html><head><title>Gear</title></head><body><h1>Landing gear</h1>"
    b"<p>The hydraulic pump must be inspected every 400 flight hours.</p>"
    b"<h1>Tyres</h1><p>Tyre pressure is checked before each flight.</p></body></html>"
)


def make_doc(content: bytes = HTML, *, version: int = 1, mime: str = "text/html") -> Document:
    return Document.create(
        tenant_id=TENANT, source="sharepoint", source_document_id="manual-1", version=version,
        source_version=f"v{version}", content=content, title="Gear manual", mime_type=mime,
        source_uri="https://example.test/manual-1", acl=ACL,
        last_modified=datetime(2026, 10, 19, tzinfo=UTC),
    )  # fmt: skip


class CountingParsers(ParserRegistry):
    def __init__(self) -> None:
        super().__init__([])
        self._inner = default_registry()
        self.calls = 0

    async def parse(self, data: bytes, mime_type: str) -> ParsedDocument:
        self.calls += 1
        return await self._inner.parse(data, mime_type)


class FlakyEmbedder:
    """Wraps an embedder; fails the next N calls with the given error."""

    def __init__(self, inner: CachingEmbedder) -> None:
        self._inner = inner
        self.failures: list[Exception] = []

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, texts: Sequence[str], *, task: EmbeddingTask) -> list[tuple[float, ...]]:
        if self.failures:
            raise self.failures.pop(0)
        return await self._inner.embed(texts, task=task)


class Rig:
    def __init__(self, enricher: object | None = None) -> None:
        self.repo = InMemoryDocumentRepository()
        self.objects = InMemoryObjectStore()
        self.parsers = CountingParsers()
        self.base = HashingEmbedder(32)
        self.embedder = FlakyEmbedder(CachingEmbedder(self.base, InMemoryEmbeddingCache()))
        self.index = InMemoryChunkIndex(model_version=self.base.model_version, dimensions=32)
        self.use_case = IngestDocument(
            documents=self.repo, assets=self.repo, objects=self.objects, parsers=self.parsers,
            embedder=self.embedder, index=self.index, enricher=enricher,  # type: ignore[arg-type]
        )  # fmt: skip

    async def status(self, doc: Document) -> Document:
        got = await self.repo.get(doc.tenant_id, doc.document_id, doc.version)
        assert got is not None
        return got


# ---- happy path ------------------------------------------------------------------------------


async def test_a_document_goes_from_bytes_to_searchable() -> None:
    rig, doc = Rig(), make_doc()
    result = await rig.use_case.ingest(doc, HTML)
    assert result.document.status is S.INDEXED and result.chunk_count >= 2
    assert len(rig.index.chunk_ids(TENANT, doc.document_id)) == result.chunk_count
    raw = await rig.objects.get(ObjectKey(TENANT, ObjectKind.RAW_DOCUMENT, doc.content_hash))
    assert raw == HTML
    assert rig.index.visible_chunk_ids(TENANT, "alice", frozenset())  # alice may see it
    assert rig.index.visible_chunk_ids(TENANT, "mallory", frozenset()) == []


async def test_images_are_stored_and_made_searchable_through_an_image_chunk() -> None:
    img = png()
    html = (
        "<html><body><h1>Gear</h1><p>See figure.</p>"
        f"<img src='{data_uri(img)}' alt='gear diagram'></body></html>"
    ).encode()

    class Enricher:
        async def enrich(self, parsed: ParsedDocument) -> Mapping[str, ImageEnrichment]:
            import hashlib

            return {
                hashlib.sha256(img).hexdigest(): ImageEnrichment(
                    ocr_text="HYD PUMP 3000 PSI",
                    description="hydraulic schematic",
                    model_version="vision-1",
                )
            }

    rig, doc = Rig(Enricher()), make_doc(html)
    result = await rig.use_case.ingest(doc, html)
    assert result.image_count == 1
    kinds = {c.kind.value for c in await rig.repo.get_chunks(TENANT, doc.document_id, 1)}
    assert kinds == {"text", "image"}
    [asset] = await rig.repo.list_assets(TENANT, doc.document_id, 1)
    assert await rig.objects.head(ObjectKey(TENANT, ObjectKind.IMAGE_ASSET, asset.sha256))


async def test_running_a_finished_document_again_does_nothing() -> None:
    rig, doc = Rig(), make_doc()
    await rig.use_case.ingest(doc, HTML)
    writes, parses = rig.index.write_calls, rig.parsers.calls
    await rig.use_case.run(TENANT, doc.document_id, 1)
    assert (rig.index.write_calls, rig.parsers.calls) == (writes, parses)


async def test_a_new_version_replaces_the_old_one_in_the_index() -> None:
    rig = Rig()
    v1 = make_doc()
    html2 = HTML.replace(b"400", b"600")
    v2 = make_doc(html2, version=2)
    await rig.use_case.ingest(v1, HTML)
    await rig.use_case.ingest(v2, html2)
    assert all(":v2:" in c for c in rig.index.chunk_ids(TENANT, v1.document_id))


# ---- failure and recovery --------------------------------------------------------------------


async def test_embedding_outage_then_retry_resumes_without_reparsing() -> None:
    rig, doc = Rig(), make_doc()
    rig.embedder.failures = [EmbeddingUnavailableError("provider says: secret text")]
    with pytest.raises(IngestionFailed) as exc:
        await rig.use_case.ingest(doc, HTML)
    assert exc.value.retryable and exc.value.reason == "embedding_unavailable"
    failed = await rig.status(doc)
    assert failed.status is S.FAILED and failed.failure_reason == "embedding_unavailable"
    assert "secret" not in (failed.failure_reason or "")
    parses = rig.parsers.calls
    result = await rig.use_case.run(TENANT, doc.document_id, 1)
    assert result.document.status is S.INDEXED
    assert rig.parsers.calls == parses  # chunks were durable: no re-parse


async def test_index_outage_then_retry_does_not_embed_again() -> None:
    rig, doc = Rig(), make_doc()
    rig.index.fail_with = SearchIndexUnavailableError("index down")
    with pytest.raises(IngestionFailed) as exc:
        await rig.use_case.ingest(doc, HTML)
    assert exc.value.retryable and exc.value.reason == "index_unavailable"
    assert exc.value.stage is S.INDEXING
    embedded = rig.base.texts_embedded
    rig.index.fail_with = None
    result = await rig.use_case.run(TENANT, doc.document_id, 1)
    assert result.document.status is S.INDEXED
    assert rig.base.texts_embedded == embedded  # vectors came from the cache


async def test_a_worker_that_died_mid_stage_is_resumed() -> None:
    rig, doc = Rig(), make_doc()
    await rig.use_case.submit(doc, HTML)
    await rig.repo.transition(TENANT, doc.document_id, 1, S.PARSING)
    await rig.repo.transition(TENANT, doc.document_id, 1, S.CHUNKING)  # ...then the process died
    result = await rig.use_case.run(TENANT, doc.document_id, 1)
    assert result.document.status is S.INDEXED and result.chunk_count >= 2


async def test_a_corrupt_document_fails_permanently_without_leaking_content() -> None:
    bad = b"%PDF-1.7 not really a pdf SECRET-CONTENT"
    rig, doc = Rig(), make_doc(bad, mime="application/pdf")
    with pytest.raises(IngestionFailed) as exc:
        await rig.use_case.ingest(doc, bad)
    assert not exc.value.retryable and exc.value.reason == "parse_error"
    stored = await rig.status(doc)
    assert stored.status is S.FAILED and "SECRET" not in (stored.failure_reason or "")
    assert rig.index.chunk_ids(TENANT, doc.document_id) == []


async def test_unsupported_type_and_empty_documents_fail_permanently() -> None:
    rig = Rig()
    odd = make_doc(b"data", mime="application/x-unknown")
    with pytest.raises(IngestionFailed) as exc:
        await rig.use_case.ingest(odd, b"data")
    assert exc.value.reason == "parse_error"
    blank = b"<html><body></body></html>"
    with pytest.raises(IngestionFailed) as exc2:
        await Rig().use_case.ingest(make_doc(blank), blank)
    assert exc2.value.reason == "no_content" and not exc2.value.retryable


async def test_permanent_provider_errors_are_not_marked_retryable() -> None:
    rig, doc = Rig(), make_doc()
    rig.embedder.failures = [EmbeddingError("bad key")]
    with pytest.raises(IngestionFailed) as exc:
        await rig.use_case.ingest(doc, HTML)
    assert exc.value.reason == "embedding_failed" and not exc.value.retryable
    rig2, doc2 = Rig(), make_doc()
    rig2.index.fail_with = SearchIndexError("mapping rejected")
    with pytest.raises(IngestionFailed) as exc2:
        await rig2.use_case.ingest(doc2, HTML)
    assert exc2.value.reason == "index_failed" and not exc2.value.retryable


async def test_an_old_version_cannot_overwrite_a_newer_one() -> None:
    rig = Rig()
    html2 = HTML.replace(b"400", b"600")
    v2 = make_doc(html2, version=2)
    await rig.use_case.ingest(v2, html2)
    v1 = make_doc()
    with pytest.raises(IngestionFailed) as exc:
        await rig.use_case.ingest(v1, HTML)
    assert exc.value.reason == "superseded" and not exc.value.retryable
    assert all(":v2:" in c for c in rig.index.chunk_ids(TENANT, v1.document_id))


async def test_unexpected_errors_are_retryable_and_recorded_generically() -> None:
    rig, doc = Rig(), make_doc()
    rig.embedder.failures = [RuntimeError("boom with /private/path")]
    with pytest.raises(IngestionFailed) as exc:
        await rig.use_case.ingest(doc, HTML)
    assert exc.value.reason == "unexpected_error" and exc.value.retryable
    assert "private" not in (await rig.status(doc)).failure_reason  # type: ignore[operator]


# ---- submit ----------------------------------------------------------------------------------


async def test_submit_is_idempotent_and_detects_conflicting_content() -> None:
    rig, doc = Rig(), make_doc()
    await rig.use_case.submit(doc, HTML)
    await rig.use_case.submit(doc, HTML)
    other = HTML + b"<!-- changed -->"
    with pytest.raises(VersionConflictError):
        await rig.use_case.submit(make_doc(other), other)  # same version, different content


async def test_submit_refuses_bytes_that_do_not_match_the_declared_hash() -> None:
    rig, doc = Rig(), make_doc()
    with pytest.raises(ContractError):
        await rig.use_case.submit(doc, b"something else entirely")


def test_embedder_and_index_must_share_a_model_version() -> None:
    rig = Rig()
    with pytest.raises(ValueError):
        IngestDocument(
            documents=rig.repo, assets=rig.repo, objects=rig.objects, parsers=rig.parsers,
            embedder=HashingEmbedder(16),
            index=InMemoryChunkIndex(model_version="other@16", dimensions=16),
        )  # fmt: skip
