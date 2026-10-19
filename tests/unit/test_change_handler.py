"""ChangeHandler: duplicates, stale and out-of-order events, ACL changes, deletes, races."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from ekip_contracts.documents import (
    Document,
    DocumentAcl,
    DocumentStatus,
    sha256_hex,
)
from ekip_contracts.events import ChangeKind, SourceChange
from ekip_contracts.object_store import ObjectKind
from ekip_contracts.search_index import SearchIndexUnavailableError
from ekip_ingestion import (
    ChangeAction,
    ChangeHandler,
    ChangeRejected,
    IngestDocument,
    IngestionFailed,
    default_registry,
)
from ekip_llm_gateway import CachingEmbedder, HashingEmbedder, InMemoryEmbeddingCache
from ekip_persistence import InMemoryDocumentRepository, InMemoryObjectStore
from ekip_search import InMemoryChunkIndex

S = DocumentStatus
A = ChangeAction
T0 = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
TENANT = "tenant-a"
ALICE = DocumentAcl("acl-1", frozenset({"alice"}), frozenset())
BOB = DocumentAcl("acl-2", frozenset({"bob"}), frozenset())


def html(price: str = "400") -> bytes:
    return (
        f"<html><head><title>Gear</title></head><body><h1>Pump</h1>"
        f"<p>Inspect the hydraulic pump every {price} flight hours without exception.</p>"
        f"</body></html>"
    ).encode()


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


class Rig:
    def __init__(self) -> None:
        self.repo = InMemoryDocumentRepository()
        self.objects = InMemoryObjectStore()
        self.base = HashingEmbedder(32)
        self.embedder = CachingEmbedder(self.base, InMemoryEmbeddingCache())
        self.index = InMemoryChunkIndex(model_version=self.base.model_version, dimensions=32)
        self.handler = ChangeHandler(documents=self.repo, index=self.index, objects=self.objects)
        self.ingest = IngestDocument(
            documents=self.repo, assets=self.repo, objects=self.objects,
            parsers=default_registry(), embedder=self.embedder, index=self.index,
        )  # fmt: skip
        self.n = 0

    async def upsert(
        self, content: bytes, minute: int, *, acl: DocumentAcl = ALICE, version: str | None = None,
        store: bool = True,
    ) -> SourceChange:  # fmt: skip
        if store:
            await self.objects.put(
                TENANT, ObjectKind.RAW_DOCUMENT, content, content_type="text/html"
            )
        self.n += 1
        return SourceChange(
            event_id=f"evt-{self.n}", tenant_id=TENANT, source="sharepoint",
            source_document_id="manual-1", kind=ChangeKind.UPSERT, occurred_at=at(minute),
            source_version=version or f"v{minute}", content_hash=sha256_hex(content),
            mime_type="text/html", title="Gear manual", source_uri="https://example.test/m",
            acl=acl,
        )  # fmt: skip

    def other(self, kind: ChangeKind, minute: int, acl: DocumentAcl | None = None) -> SourceChange:
        self.n += 1
        return SourceChange(
            event_id=f"evt-{self.n}", tenant_id=TENANT, source="sharepoint",
            source_document_id="manual-1", kind=kind, occurred_at=at(minute), acl=acl,
        )  # fmt: skip

    async def apply(self, content: bytes, minute: int, **kw: object) -> ChangeAction:
        """UPSERT then, if asked, run ingestion."""
        out = await self.handler.handle(await self.upsert(content, minute, **kw))  # type: ignore[arg-type]
        if out.needs_ingest and out.version:
            await self.ingest.run(TENANT, out.document_id, out.version)
        return out.action

    def visible(self, user: str) -> list[str]:
        return self.index.visible_chunk_ids(TENANT, user, frozenset())

    async def versions(self) -> list[Document]:
        ev = await self.upsert(html(), 0, store=False)
        return await self.repo.list_versions(TENANT, ev.document_id)


# ---- UPSERT ----------------------------------------------------------------------------------


async def test_first_upsert_registers_version_1_and_ingestion_makes_it_searchable() -> None:
    rig = Rig()
    assert await rig.apply(html(), 1) is A.INGEST
    [v1] = await rig.versions()
    assert v1.version == 1 and v1.status is S.INDEXED and v1.acl_effective_at == at(1)
    assert rig.visible("alice") and rig.visible("bob") == []


async def test_redelivery_creates_nothing() -> None:
    rig = Rig()
    ev = await rig.upsert(html(), 1)
    first = await rig.handler.handle(ev)
    await rig.ingest.run(TENANT, first.document_id, 1)
    for _ in range(3):
        assert (await rig.handler.handle(ev)).action is A.DUPLICATE
    assert len(await rig.versions()) == 1


async def test_metadata_only_change_is_not_a_new_version_and_costs_no_embeddings() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    embedded = rig.base.texts_embedded
    assert await rig.apply(html(), 2, version="etag-changed") is A.DUPLICATE
    assert len(await rig.versions()) == 1 and rig.base.texts_embedded == embedded


async def test_content_change_registers_the_next_version_and_replaces_the_old_chunks() -> None:
    rig = Rig()
    await rig.apply(html("400"), 1)
    assert await rig.apply(html("600"), 2) is A.INGEST
    assert [v.version for v in await rig.versions()] == [1, 2]
    assert all(":v2:" in c for c in rig.visible("alice"))


async def test_an_older_event_arriving_late_is_ignored() -> None:
    rig = Rig()
    await rig.apply(html("600"), 5)
    assert await rig.apply(html("400"), 2) is A.IGNORED_STALE  # a backfill snapshot from earlier
    assert len(await rig.versions()) == 1


async def test_a_failed_version_is_retried_by_a_redelivered_event() -> None:
    rig = Rig()
    ev = await rig.upsert(html(), 1)
    out = await rig.handler.handle(ev)
    rig.index.fail_with = SearchIndexUnavailableError("down")
    with pytest.raises(IngestionFailed):
        await rig.ingest.run(TENANT, out.document_id, 1)
    rig.index.fail_with = None
    again = await rig.handler.handle(ev)
    assert again.action is A.INGEST and again.version == 1  # same version, not a new one
    await rig.ingest.run(TENANT, out.document_id, 1)
    assert rig.visible("alice")


async def test_missing_content_is_reported_not_registered() -> None:
    rig = Rig()
    ev = await rig.upsert(html(), 1, store=False)
    with pytest.raises(ChangeRejected) as exc:
        await rig.handler.handle(ev)
    assert exc.value.reason == "content_missing" and exc.value.retryable
    assert await rig.versions() == []


async def test_concurrent_different_upserts_never_collide_on_a_version_number() -> None:
    rig = Rig()
    e1, e2 = await rig.upsert(html("100"), 1), await rig.upsert(html("200"), 2)
    outcomes = await asyncio.gather(rig.handler.handle(e1), rig.handler.handle(e2))
    versions = [v.version for v in await rig.versions()]
    assert versions == sorted(set(versions)) and len(versions) >= 1
    assert all(o.action in (A.INGEST, A.IGNORED_STALE) for o in outcomes)


# ---- ACL -------------------------------------------------------------------------------------


async def test_acl_change_takes_effect_in_repository_and_index_without_reembedding() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    embedded = rig.base.texts_embedded
    out = await rig.handler.handle(rig.other(ChangeKind.ACL_CHANGE, 3, BOB))
    assert out.action is A.ACL_APPLIED
    assert rig.visible("alice") == [] and rig.visible("bob")  # revoked and granted at once
    [v] = await rig.versions()
    assert v.acl == BOB and v.acl_effective_at == at(3)
    assert rig.base.texts_embedded == embedded


async def test_an_older_acl_change_cannot_undo_a_newer_one() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    await rig.handler.handle(rig.other(ChangeKind.ACL_CHANGE, 5, BOB))
    out = await rig.handler.handle(rig.other(ChangeKind.ACL_CHANGE, 3, ALICE))
    assert out.action is A.IGNORED_STALE
    assert rig.visible("alice") == [] and rig.visible("bob")


async def test_an_older_snapshot_with_new_content_keeps_the_newer_acl() -> None:
    """Backfill replays an old snapshot after a live permission change: content is accepted, the
    permissions of the snapshot are NOT."""
    rig = Rig()
    await rig.apply(html("400"), 1)
    await rig.handler.handle(rig.other(ChangeKind.ACL_CHANGE, 5, BOB))
    # occurred_at (3) is newer than the content (1) but older than the ACL change (5)
    assert await rig.apply(html("600"), 3, acl=ALICE) is A.INGEST
    assert rig.visible("alice") == [] and rig.visible("bob")


async def test_acl_change_for_an_unknown_document_is_ignored() -> None:
    rig = Rig()
    out = await rig.handler.handle(rig.other(ChangeKind.ACL_CHANGE, 3, BOB))
    assert out.action is A.IGNORED_UNKNOWN


async def test_acl_retry_after_a_failed_index_step_heals() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    ev = rig.other(ChangeKind.ACL_CHANGE, 3, BOB)

    original = rig.index.update_acl

    async def boom(*_: object, **__: object) -> int:
        raise SearchIndexUnavailableError("down")

    rig.index.update_acl = boom  # type: ignore[method-assign]
    with pytest.raises(SearchIndexUnavailableError):
        await rig.handler.handle(ev)
    rig.index.update_acl = original  # type: ignore[method-assign]
    assert (await rig.handler.handle(ev)).action is A.ACL_APPLIED  # same event, retried
    assert rig.visible("alice") == [] and rig.visible("bob")


# ---- DELETE ----------------------------------------------------------------------------------


async def test_delete_removes_the_document_everywhere() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    out = await rig.handler.handle(rig.other(ChangeKind.DELETE, 4))
    assert out.action is A.DELETED and rig.visible("alice") == []
    assert {v.status for v in await rig.versions()} == {S.DELETED}


async def test_duplicate_delete_still_repeats_the_index_step() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    ev = rig.other(ChangeKind.DELETE, 4)
    # first attempt: repository tombstoned, index step fails (crash between the two steps)
    original = rig.index.delete_document

    async def boom(*_: object) -> None:
        raise SearchIndexUnavailableError("down")

    rig.index.delete_document = boom  # type: ignore[assignment,method-assign]
    with pytest.raises(SearchIndexUnavailableError):
        await rig.handler.handle(ev)
    assert rig.visible("alice")  # still searchable...
    rig.index.delete_document = original  # type: ignore[method-assign]
    assert (await rig.handler.handle(ev)).action is A.DUPLICATE  # ...until the retry heals it
    assert rig.visible("alice") == []


async def test_a_delete_older_than_the_current_content_is_ignored() -> None:
    rig = Rig()
    await rig.apply(html(), 5)
    assert (await rig.handler.handle(rig.other(ChangeKind.DELETE, 2))).action is A.IGNORED_STALE
    assert rig.visible("alice")


async def test_an_old_snapshot_cannot_resurrect_a_deleted_document() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    await rig.handler.handle(rig.other(ChangeKind.DELETE, 4))
    assert await rig.apply(html("600"), 3) is A.IGNORED_STALE  # older than the tombstone
    assert rig.visible("alice") == []


async def test_a_genuinely_newer_upsert_after_a_delete_creates_a_new_version() -> None:
    rig = Rig()
    await rig.apply(html(), 1)
    await rig.handler.handle(rig.other(ChangeKind.DELETE, 4))
    assert await rig.apply(html("600"), 8) is A.INGEST
    assert [v.version for v in await rig.versions()] == [1, 2]
    assert rig.visible("alice") and all(":v2:" in c for c in rig.visible("alice"))


async def test_delete_for_an_unknown_document_is_ignored() -> None:
    assert (
        await Rig().handler.handle(Rig().other(ChangeKind.DELETE, 1))
    ).action is A.IGNORED_UNKNOWN


# ---- races with an in-flight ingestion --------------------------------------------------------


class MidPublish(InMemoryChunkIndex):
    """Runs a hook between "chunks are in the index" and "ingestion declares success"."""

    hook: Callable[[], Awaitable[None]] | None = None

    async def replace_document(self, document: Document, entries):  # type: ignore[no-untyped-def]
        n = await super().replace_document(document, entries)
        if self.hook is not None:
            hook, self.hook = self.hook, None
            await hook()
        return n


def racing_rig() -> tuple[Rig, MidPublish]:
    rig = Rig()
    idx = MidPublish(model_version=rig.base.model_version, dimensions=32)
    rig.index = idx
    rig.handler = ChangeHandler(documents=rig.repo, index=idx, objects=rig.objects)
    rig.ingest = IngestDocument(
        documents=rig.repo, assets=rig.repo, objects=rig.objects, parsers=default_registry(),
        embedder=rig.embedder, index=idx,
    )  # fmt: skip
    return rig, idx


async def test_a_delete_that_lands_during_publication_does_not_leave_chunks_behind() -> None:
    rig, idx = racing_rig()
    out = await rig.handler.handle(await rig.upsert(html(), 1))

    async def delete_now() -> None:
        # the handler updates the repository, then removes from the index, then ingestion resumes
        await rig.handler.handle(rig.other(ChangeKind.DELETE, 4))

    idx.hook = delete_now
    with pytest.raises(IngestionFailed) as exc:
        await rig.ingest.run(TENANT, out.document_id, 1)
    assert exc.value.reason == "deleted" and not exc.value.retryable
    assert rig.visible("alice") == []


async def test_an_acl_change_that_lands_during_publication_is_not_overwritten() -> None:
    rig, idx = racing_rig()
    out = await rig.handler.handle(await rig.upsert(html(), 1))

    async def revoke_now() -> None:
        await rig.handler.handle(rig.other(ChangeKind.ACL_CHANGE, 4, BOB))

    idx.hook = revoke_now
    await rig.ingest.run(TENANT, out.document_id, 1)
    assert rig.visible("alice") == [] and rig.visible("bob")
