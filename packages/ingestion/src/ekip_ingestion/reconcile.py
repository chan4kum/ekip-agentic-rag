"""Reconciler: make our state agree with the source, whatever the event stream missed.

Webhooks drop, change feeds have retention limits, consumers get DLQ'd. So periodically we ask the
source "what exists right now?" and diff that against what we hold, per tenant and source:

  in source, unknown to us            -> UPSERT   (missed create)
  in source, source_version differs   -> UPSERT   (missed update)
  in source, ACL differs              -> ACL_CHANGE (missed permission change; no re-embedding)
  in source, we hold a tombstone      -> UPSERT only if the source copy is NEWER than the tombstone
  held by us, absent from the source  -> DELETE   (missed delete / "ghost")

The reconciler never changes state itself. It publishes the same SourceChange events as a live
connector (on the backfill lane; DELETE and ACL_CHANGE are routed to the live lane by
`ChangeTopics`), so ordering, idempotency and safety rules live in exactly one place: ChangeHandler.

Safety, because "absent from the listing" means "delete":
- The inventory must be complete. Any listing error aborts the run BEFORE a single delete.
- A mass-delete guard: if the run would delete more than `max_delete_fraction` of what we hold
  (and at least `min_deletes_for_guard`), deletes are withheld and reported, for a human to look at.
  A broken connector that returns half a listing must not wipe half the knowledge base.
- Fail closed on conflicts: a tombstone newer than the source copy is not resurrected.

Memory: the diff holds one Document per known document and one id per listed document. Fine for
millions of rows per run; shard by prefix/partition if a single tenant+source grows beyond that.
"""

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from ekip_contracts.document_store import DocumentRepository
from ekip_contracts.documents import Document, DocumentStatus, sha256_hex
from ekip_contracts.events import ChangeKind, Lane, SourceChange
from ekip_contracts.messaging import ProcessingFailure
from ekip_contracts.object_store import ObjectKind, ObjectStore
from ekip_contracts.source import SourceConnector, SourceError, SourceItem, SourceItemGoneError

from ekip_ingestion.consumer import ChangePublisher


@dataclass(frozen=True)
class ReconcileReport:
    listed: int = 0
    upserts: int = 0
    acl_changes: int = 0
    deletes: int = 0
    unchanged: int = 0
    skipped_tombstoned: int = 0  # source still lists it, but our tombstone is newer: not revived
    vanished: int = 0  # listed, but gone by the time we fetched it (next run deletes it)
    fetch_failed: int = 0
    deletes_withheld: int = 0  # mass-delete guard tripped: this many deletes were NOT sent

    @property
    def guard_tripped(self) -> bool:
        return self.deletes_withheld > 0


class Reconciler:
    def __init__(
        self,
        *,
        connector: SourceConnector,
        documents: DocumentRepository,
        objects: ObjectStore,
        publisher: ChangePublisher,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_delete_fraction: float = 0.3,
        min_deletes_for_guard: int = 10,
        fetch_concurrency: int = 8,
    ) -> None:
        if not 0 < max_delete_fraction <= 1:
            raise ValueError("max_delete_fraction must be in (0, 1]")
        self._connector = connector
        self._docs = documents
        self._objects = objects
        self._publisher = publisher
        self._clock = clock
        self._max_delete_fraction = max_delete_fraction
        self._min_deletes = min_deletes_for_guard
        self._fetch_slots = asyncio.Semaphore(fetch_concurrency)

    async def run(self, tenant_id: str) -> ReconcileReport:
        source = self._connector.source
        ours = {d.source_document_id: d for d in await self._docs.list_latest(tenant_id, source)}

        listed: list[SourceItem] = []
        try:
            async for item in self._connector.inventory(tenant_id):
                listed.append(item)
        except SourceError:
            # Incomplete listing: sending deletes now would be destructive. Retry the whole run.
            raise ProcessingFailure("source_unavailable", retryable=True) from None
        seen = {i.source_document_id for i in listed}

        counts = {
            "unchanged": 0, "skipped_tombstoned": 0, "vanished": 0, "fetch_failed": 0,
            "acl_changes": 0, "upserts": 0,
        }  # fmt: skip
        to_upsert: list[SourceItem] = []
        for item in listed:
            known = ours.get(item.source_document_id)
            action = self._decide(item, known)
            if action == "upsert":
                to_upsert.append(item)
            elif action == "acl":
                await self._publish_acl(tenant_id, item)
                counts["acl_changes"] += 1
            elif action == "tombstoned":
                counts["skipped_tombstoned"] += 1
            else:
                counts["unchanged"] += 1

        results = await asyncio.gather(*(self._upsert(tenant_id, i) for i in to_upsert))
        for outcome in results:
            counts[outcome] += 1

        ghosts = [
            d
            for sid, d in ours.items()
            if sid not in seen and d.status is not DocumentStatus.DELETED
        ]
        live_known = sum(1 for d in ours.values() if d.status is not DocumentStatus.DELETED)
        withheld = 0
        if (
            len(ghosts) >= self._min_deletes
            and len(ghosts) > self._max_delete_fraction * live_known
        ):
            withheld, ghosts = len(ghosts), []
        now = self._clock()
        for ghost in ghosts:
            await self._publisher.publish(
                SourceChange(
                    event_id=uuid.uuid4().hex,
                    tenant_id=tenant_id,
                    source=source,
                    source_document_id=ghost.source_document_id,
                    kind=ChangeKind.DELETE,
                    occurred_at=now,
                    lane=Lane.BACKFILL,
                )  # fmt: skip
            )

        return ReconcileReport(
            listed=len(listed), deletes=len(ghosts), deletes_withheld=withheld, **counts
        )

    # ---- decisions ---------------------------------------------------------------------------

    @staticmethod
    def _decide(item: SourceItem, known: Document | None) -> str:
        if known is None:
            return "upsert"
        if known.status is DocumentStatus.DELETED:
            if known.deleted_at is not None and item.last_modified <= known.deleted_at:
                return "tombstoned"
            return "upsert"  # recreated at the source after we deleted it
        if item.source_version != known.source_version:
            return "upsert"
        if item.acl != known.acl and item.acl_effective_at >= known.acl_effective_at:
            return "acl"
        return "unchanged"

    # ---- actions -----------------------------------------------------------------------------

    async def _upsert(self, tenant_id: str, item: SourceItem) -> str:
        async with self._fetch_slots:
            try:
                data = await self._connector.fetch(tenant_id, item.source_document_id)
            except SourceItemGoneError:
                return "vanished"
            except SourceError:
                return "fetch_failed"
        stored = await self._objects.put(
            tenant_id, ObjectKind.RAW_DOCUMENT, data, content_type=item.mime_type
        )
        assert stored.key.sha256 == sha256_hex(data)  # noqa: S101 - adapter contract
        await self._publisher.publish(
            SourceChange(
                event_id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                source=self._connector.source,
                source_document_id=item.source_document_id,
                kind=ChangeKind.UPSERT,
                occurred_at=item.last_modified,
                lane=Lane.BACKFILL,
                source_version=item.source_version,
                content_hash=stored.key.sha256,
                mime_type=item.mime_type,
                title=item.title,
                source_uri=item.source_uri,
                acl=item.acl,
            )  # fmt: skip
        )
        return "upserts"

    async def _publish_acl(self, tenant_id: str, item: SourceItem) -> None:
        await self._publisher.publish(
            SourceChange(
                event_id=uuid.uuid4().hex,
                tenant_id=tenant_id,
                source=self._connector.source,
                source_document_id=item.source_document_id,
                kind=ChangeKind.ACL_CHANGE,
                occurred_at=item.acl_effective_at,
                lane=Lane.BACKFILL,
                acl=item.acl,
            )  # fmt: skip
        )
