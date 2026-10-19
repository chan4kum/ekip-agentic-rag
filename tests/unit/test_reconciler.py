"""Reconciler: heal what the event stream missed, and refuse to be destructive on bad data."""

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta

import pytest
from ekip_contracts.documents import DocumentAcl, DocumentStatus
from ekip_contracts.events import ChangeKind, Lane, SourceChange
from ekip_contracts.messaging import ChangeTopics, ProcessingFailure
from ekip_contracts.object_store import ObjectKey, ObjectKind
from ekip_contracts.source import SourceError, SourceItem, SourceItemGoneError
from ekip_ingestion import (
    ChangeHandler,
    ChangeProcessor,
    ChangePublisher,
    IngestDocument,
    InlineIngestTrigger,
    Reconciler,
    default_registry,
)
from ekip_llm_gateway import CachingEmbedder, HashingEmbedder, InMemoryEmbeddingCache
from ekip_persistence import InMemoryDocumentRepository, InMemoryObjectStore
from ekip_search import InMemoryChunkIndex

S = DocumentStatus
T0 = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
NOW = T0 + timedelta(hours=5)
TENANT = "tenant-a"
SOURCE = "sharepoint"
ALICE = DocumentAcl("acl-1", frozenset({"alice"}), frozenset())
BOB = DocumentAcl("acl-2", frozenset({"bob"}), frozenset())
TOPICS = ChangeTopics()


def at(minutes: int) -> datetime:
    return T0 + timedelta(minutes=minutes)


def html(topic: str) -> bytes:
    return (
        f"<html><head><title>{topic}</title></head><body><h1>{topic}</h1>"
        f"<p>The {topic} procedure must be followed exactly as written, every single time.</p>"
        f"</body></html>"
    ).encode()


class FakeSource:
    """A source with a mutable inventory, and switches for the ways real sources misbehave."""

    source = SOURCE

    def __init__(self) -> None:
        self.docs: dict[str, tuple[SourceItem, bytes]] = {}
        self.fail_listing_after: int | None = None
        self.gone: set[str] = set()
        self.broken: set[str] = set()

    def put(
        self, sid: str, topic: str, *, minute: int = 1, version: str | None = None,
        acl: DocumentAcl = ALICE, acl_minute: int | None = None,
    ) -> None:  # fmt: skip
        item = SourceItem(
            source_document_id=sid, source_version=version or f"v{minute}",
            last_modified=at(minute), acl=acl, mime_type="text/html", title=topic,
            source_uri=f"https://example.test/{sid}",
            acl_as_of=at(acl_minute) if acl_minute is not None else None,
        )  # fmt: skip
        self.docs[sid] = (item, html(topic))

    async def inventory(self, tenant_id: str) -> AsyncIterator[SourceItem]:
        for n, (item, _) in enumerate(list(self.docs.values())):
            if self.fail_listing_after is not None and n >= self.fail_listing_after:
                raise SourceError("throttled")
            yield item

    async def fetch(self, tenant_id: str, source_document_id: str) -> bytes:
        if source_document_id in self.gone:
            raise SourceItemGoneError(source_document_id)
        if source_document_id in self.broken:
            raise SourceError("boom")
        return self.docs[source_document_id][1]


class Recorder:
    def __init__(self) -> None:
        self.sent: list[tuple[str, SourceChange]] = []

    async def publish(
        self, topic: str, *, key: str, value: bytes, headers: Mapping[str, str] | None = None
    ) -> None:
        self.sent.append((topic, SourceChange.from_json(value)))

    async def close(self) -> None: ...

    def kinds(self) -> list[ChangeKind]:
        return [c.kind for _, c in self.sent]


class Rig:
    def __init__(self, **kw: object) -> None:
        self.source = FakeSource()
        self.repo = InMemoryDocumentRepository()
        self.objects = InMemoryObjectStore()
        self.out = Recorder()
        self.base = HashingEmbedder(32)
        self.index = InMemoryChunkIndex(model_version=self.base.model_version, dimensions=32)
        self.ingest = IngestDocument(
            documents=self.repo, assets=self.repo, objects=self.objects,
            parsers=default_registry(),
            embedder=CachingEmbedder(self.base, InMemoryEmbeddingCache()), index=self.index,
        )  # fmt: skip
        self.processor = ChangeProcessor(
            ChangeHandler(documents=self.repo, index=self.index, objects=self.objects),
            InlineIngestTrigger(self.ingest),
        )
        self.reconciler = Reconciler(
            connector=self.source, documents=self.repo, objects=self.objects,
            publisher=ChangePublisher(self.out, TOPICS), clock=lambda: NOW, **kw,  # type: ignore[arg-type]
        )  # fmt: skip

    async def reconcile(self, *, apply: bool = True) -> tuple[object, list[SourceChange]]:
        self.out.sent.clear()
        report = await self.reconciler.run(TENANT)
        events = [c for _, c in self.out.sent]
        if apply:  # deliver them like the consumer would
            for change in events:
                await self.processor.process(change)
        return report, events

    def visible(self, user: str) -> list[str]:
        return self.index.visible_chunk_ids(TENANT, user, frozenset())


# ---- diffing ---------------------------------------------------------------------------------


async def test_everything_missing_is_created_through_the_backfill_lane() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    rig.source.put("b", "Tyres")
    report, events = await rig.reconcile(apply=False)
    assert (report.listed, report.upserts) == (2, 2)  # type: ignore[attr-defined]
    assert {c.kind for c in events} == {ChangeKind.UPSERT}
    assert {topic for topic, _ in rig.out.sent} == {TOPICS.backfill}
    assert all(c.lane is Lane.BACKFILL for c in events)
    # the bytes were stored before the event was sent, so the handler can accept it
    for change in events:
        key = ObjectKey(TENANT, ObjectKind.RAW_DOCUMENT, change.content_hash or "")
        assert await rig.objects.head(key) is not None


async def test_an_unchanged_source_produces_no_events() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    await rig.reconcile()
    report, events = await rig.reconcile()
    assert events == [] and report.unchanged == 1  # type: ignore[attr-defined]


async def test_a_missed_update_is_detected_by_source_version() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    await rig.reconcile()
    rig.source.put("a", "Pumps revised", minute=30)
    _, events = await rig.reconcile()
    assert [c.kind for c in events] == [ChangeKind.UPSERT]
    [doc] = await rig.repo.list_latest(TENANT, SOURCE)
    assert doc.version == 2 and doc.status is S.INDEXED


async def test_a_missed_permission_change_is_applied_without_reembedding() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    await rig.reconcile()
    embedded = rig.base.texts_embedded
    rig.source.put("a", "Pumps", acl=BOB, acl_minute=40)  # same version, different ACL
    _, events = await rig.reconcile()
    assert [c.kind for c in events] == [ChangeKind.ACL_CHANGE]
    assert rig.visible("bob") and rig.visible("alice") == []
    assert rig.base.texts_embedded == embedded
    assert rig.out.sent[0][0] == TOPICS.live  # revocations never wait behind a backfill


async def test_an_acl_snapshot_older_than_ours_is_not_applied() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    await rig.reconcile()
    doc_id = (await rig.repo.list_latest(TENANT, SOURCE))[0].document_id
    newer = SourceChange(
        event_id="live-1", tenant_id=TENANT, source=SOURCE, source_document_id="a",
        kind=ChangeKind.ACL_CHANGE, occurred_at=at(50), acl=BOB,
    )  # fmt: skip
    await rig.processor.process(newer)  # a live permission change we already applied
    rig.source.put("a", "Pumps", acl=ALICE, acl_minute=10)  # but the listing was read earlier
    _, events = await rig.reconcile()
    assert events == []
    assert (await rig.repo.get(TENANT, doc_id, 1)).acl == BOB  # type: ignore[union-attr]


async def test_a_missed_delete_removes_the_document() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    rig.source.put("b", "Tyres")
    await rig.reconcile()
    del rig.source.docs["b"]
    report, events = await rig.reconcile()
    assert [c.kind for c in events] == [ChangeKind.DELETE] and report.deletes == 1  # type: ignore[attr-defined]
    assert events[0].occurred_at == NOW and rig.out.sent[0][0] == TOPICS.live
    assert all(":v" in c and "doc" in c for c in rig.visible("alice"))
    docs = {d.source_document_id: d for d in await rig.repo.list_latest(TENANT, SOURCE)}
    assert docs["b"].status is S.DELETED and docs["a"].status is S.INDEXED


async def test_a_deleted_document_is_not_deleted_again() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    await rig.reconcile()
    del rig.source.docs["a"]
    await rig.reconcile()
    _, events = await rig.reconcile()
    assert events == []


# ---- tombstones ------------------------------------------------------------------------------


async def test_a_tombstone_newer_than_the_source_copy_is_not_revived() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps", minute=1)
    await rig.reconcile()
    doc_id = (await rig.repo.list_latest(TENANT, SOURCE))[0].document_id
    await rig.repo.mark_deleted(TENANT, doc_id, at(100))  # deleted after the source copy
    report, events = await rig.reconcile()
    assert events == [] and report.skipped_tombstoned == 1  # type: ignore[attr-defined]


async def test_a_document_recreated_at_the_source_after_deletion_comes_back() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps", minute=1)
    await rig.reconcile()
    doc_id = (await rig.repo.list_latest(TENANT, SOURCE))[0].document_id
    await rig.repo.mark_deleted(TENANT, doc_id, at(50))
    rig.source.put("a", "Pumps again", minute=90)
    _, events = await rig.reconcile()
    assert [c.kind for c in events] == [ChangeKind.UPSERT]
    [doc] = await rig.repo.list_latest(TENANT, SOURCE)
    assert doc.status is S.INDEXED and doc.version == 2


# ---- safety ----------------------------------------------------------------------------------


async def test_an_incomplete_listing_aborts_before_any_event_is_sent() -> None:
    rig = Rig()
    for n in range(5):
        rig.source.put(f"d{n}", f"Topic {n}")
    await rig.reconcile()
    rig.source.fail_listing_after = 2  # the source throttles half way through
    rig.source.put("d0", "Topic 0 edited", minute=60)
    with pytest.raises(ProcessingFailure) as caught:
        await rig.reconcile()
    assert caught.value.reason == "source_unavailable" and caught.value.retryable
    assert rig.out.sent == []  # not even the upserts: nothing is half-applied


async def test_the_mass_delete_guard_withholds_suspicious_deletes() -> None:
    rig = Rig()
    for n in range(20):
        rig.source.put(f"d{n}", f"Topic {n}")
    await rig.reconcile()
    keep = dict(list(rig.source.docs.items())[:5])
    rig.source.docs = keep  # source "loses" 15 of 20 (e.g. a broken connector)
    report, events = await rig.reconcile()
    assert events == [] and report.deletes_withheld == 15 and report.guard_tripped  # type: ignore[attr-defined]
    assert len(rig.visible("alice")) >= 20  # nothing was removed


async def test_small_deletions_are_not_blocked_by_the_guard() -> None:
    rig = Rig()
    for n in range(3):
        rig.source.put(f"d{n}", f"Topic {n}")
    await rig.reconcile()
    rig.source.docs = {}
    report, events = await rig.reconcile()
    assert len(events) == 3 and report.deletes == 3 and not report.guard_tripped  # type: ignore[attr-defined]


async def test_fetch_problems_are_counted_and_do_not_stop_the_run_or_delete_anything() -> None:
    rig = Rig()
    rig.source.put("ok", "Fine")
    rig.source.put("gone", "Vanishing")
    rig.source.put("broken", "Broken")
    rig.source.gone.add("gone")
    rig.source.broken.add("broken")
    report, events = await rig.reconcile()
    assert (report.upserts, report.vanished, report.fetch_failed) == (1, 1, 1)  # type: ignore[attr-defined]
    assert [c.source_document_id for c in events] == ["ok"]


async def test_other_tenants_and_other_sources_are_never_touched() -> None:
    rig = Rig()
    rig.source.put("a", "Pumps")
    await rig.reconcile()
    other_tenant = "tenant-b"
    # a document of another tenant, and another source of the same tenant
    from ekip_contracts.documents import Document

    for tenant, source in ((other_tenant, SOURCE), (TENANT, "confluence")):
        await rig.repo.register(
            Document.create(
                tenant_id=tenant,
                source=source,
                source_document_id="x",
                version=1,
                source_version="v1",
                content=b"x",
                title="X",
                mime_type="text/html",
                source_uri="https://example.test/x",
                acl=ALICE,
                last_modified=T0,
            )  # fmt: skip
        )
    rig.source.docs = {}  # the source is now empty
    _, events = await rig.reconcile()
    assert [c.source_document_id for c in events] == ["a"]  # only OUR tenant+source


# ---- convergence -----------------------------------------------------------------------------


async def test_after_missed_events_the_system_converges_and_then_goes_quiet() -> None:
    rig = Rig()
    for sid, topic in (("a", "Pumps"), ("b", "Tyres"), ("c", "Brakes"), ("d", "Flaps")):
        rig.source.put(sid, topic)
    await rig.reconcile()
    # while the event stream was down: b edited, c deleted, d re-permissioned, e created
    rig.source.put("b", "Tyres revised", minute=45)
    del rig.source.docs["c"]
    rig.source.put("d", "Flaps", acl=BOB, acl_minute=46)
    rig.source.put("e", "Doors", minute=47)
    report, events = await rig.reconcile()
    assert sorted(c.kind.value for c in events) == ["acl_change", "delete", "upsert", "upsert"]
    latest = {d.source_document_id: d for d in await rig.repo.list_latest(TENANT, SOURCE)}
    assert {k: v.status for k, v in latest.items()} == {
        "a": S.INDEXED, "b": S.INDEXED, "c": S.DELETED, "d": S.INDEXED, "e": S.INDEXED,
    }  # fmt: skip
    assert latest["b"].version == 2 and latest["d"].acl == BOB
    again, quiet = await rig.reconcile()
    assert quiet == [] and again.unchanged == 4  # type: ignore[attr-defined]
    assert len(rig.visible("bob")) > 0 and not any(
        c.startswith(latest["c"].document_id) for c in rig.visible("alice")
    )


def test_the_guard_fraction_is_validated() -> None:
    with pytest.raises(ValueError, match="max_delete_fraction"):
        Rig(max_delete_fraction=0)
