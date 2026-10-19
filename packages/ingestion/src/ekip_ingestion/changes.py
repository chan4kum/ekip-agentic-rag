"""ChangeHandler: turn a SourceChange event into the right action, safely.

Events are delivered at least once and can arrive out of order (a backfill replays old snapshots
while live updates flow). The handler therefore decides from STATE and SOURCE TIMES, never from
arrival order:

  UPSERT      duplicate content -> no new version (metadata-only changes included)
              older than what we hold, or older than a tombstone -> ignored (stale)
              otherwise -> register the next version (to be ingested)
  ACL_CHANGE  applied unless a newer ACL is already in place; no re-embedding
  DELETE      tombstones every version and removes the chunks from the index

Ordering of side effects: the repository is updated FIRST, then the index. Both steps are
idempotent and a failure in the second simply makes the event retry. `IngestDocument` re-checks
the repository after it publishes, which closes the race with an in-flight ingestion (see
pipeline.py). A DELETE or ACL_CHANGE also repeats its index step on a duplicate delivery, so a
crash between the two steps heals on retry.

Deliberate limits (see learning log): a DELETE for a document we have never seen leaves no
tombstone, so an older UPSERT arriving later would create it; reconciliation (Gate 4.7) removes
such ghosts. Erasing stored payloads (raw bytes, assets) on delete is a separate retention job.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ekip_contracts.document_store import DocumentRepository, VersionConflictError
from ekip_contracts.documents import Document, DocumentAcl, DocumentStatus
from ekip_contracts.events import ChangeKind, SourceChange
from ekip_contracts.messaging import ProcessingFailure
from ekip_contracts.object_store import ObjectKey, ObjectKind, ObjectStore
from ekip_contracts.search_index import ChunkIndex

S = DocumentStatus
_MAX_REGISTER_ATTEMPTS = 3


class ChangeAction(StrEnum):
    INGEST = "ingest"  # a version needs processing (new, or a failed one being retried)
    DUPLICATE = "duplicate"  # nothing to do: already applied or already in progress
    IGNORED_STALE = "ignored_stale"  # older than the state we hold
    IGNORED_UNKNOWN = "ignored_unknown"  # ACL/DELETE for a document we have never seen
    ACL_APPLIED = "acl_applied"
    DELETED = "deleted"


@dataclass(frozen=True)
class ChangeOutcome:
    action: ChangeAction
    document_id: str
    version: int | None = None

    @property
    def needs_ingest(self) -> bool:
        return self.action is ChangeAction.INGEST


class ChangeRejected(ProcessingFailure):
    """The event cannot be applied right now. `retryable` says whether trying again can help."""


class ChangeHandler:
    def __init__(
        self, *, documents: DocumentRepository, index: ChunkIndex, objects: ObjectStore
    ) -> None:
        self._docs = documents
        self._index = index
        self._objects = objects

    async def handle(self, change: SourceChange) -> ChangeOutcome:
        if change.kind is ChangeKind.UPSERT:
            return await self._upsert(change)
        if change.kind is ChangeKind.ACL_CHANGE:
            return await self._acl_change(change)
        return await self._delete(change)

    # ---- UPSERT -----------------------------------------------------------------------------

    async def _upsert(self, change: SourceChange) -> ChangeOutcome:
        acl_in, content_hash = _required(change.acl), _required(change.content_hash)
        doc_id = change.document_id
        for _ in range(_MAX_REGISTER_ATTEMPTS):
            versions = await self._docs.list_versions(change.tenant_id, doc_id)
            tombstone = _tombstone(versions)
            if tombstone is not None and change.occurred_at <= tombstone:
                return ChangeOutcome(ChangeAction.IGNORED_STALE, doc_id)  # no resurrection
            live = [v for v in versions if v.status is not S.DELETED]
            current = max(live, key=lambda v: v.version) if live else None
            if current is not None and change.occurred_at < current.last_modified:
                return ChangeOutcome(ChangeAction.IGNORED_STALE, doc_id, current.version)

            if current is not None and current.content_hash == content_hash:
                # Same bytes: never a new version. A newer ACL carried by the event still applies.
                if change.occurred_at >= current.acl_effective_at and acl_in != current.acl:
                    await self._apply_acl(change, acl_in)
                action = (
                    ChangeAction.INGEST if current.status is S.FAILED else ChangeAction.DUPLICATE
                )
                return ChangeOutcome(action, doc_id, current.version)

            if (
                await self._objects.head(
                    ObjectKey(change.tenant_id, ObjectKind.RAW_DOCUMENT, content_hash)
                )
                is None
            ):
                raise ChangeRejected("content_missing", retryable=True)

            # A newer ACL that arrived separately must survive an older snapshot's ACL.
            if current is not None and change.occurred_at < current.acl_effective_at:
                acl, acl_as_of = current.acl, current.acl_effective_at
            else:
                acl, acl_as_of = acl_in, change.occurred_at
            version = max((v.version for v in versions), default=0) + 1
            candidate = Document(
                tenant_id=change.tenant_id,
                source=change.source,
                source_document_id=change.source_document_id,
                document_id=doc_id,
                version=version,
                source_version=change.source_version or "",
                content_hash=content_hash,
                title=change.title or "",
                mime_type=change.mime_type or "",
                source_uri=change.source_uri or "",
                acl=acl,
                last_modified=change.occurred_at,
                acl_as_of=acl_as_of,
            )
            try:
                await self._docs.register(candidate)
            except VersionConflictError:
                continue  # another worker took this version number: re-read and decide again
            return ChangeOutcome(ChangeAction.INGEST, doc_id, version)
        raise ChangeRejected("version_conflict", retryable=True)

    # ---- ACL_CHANGE -------------------------------------------------------------------------

    async def _acl_change(self, change: SourceChange) -> ChangeOutcome:
        new_acl = _required(change.acl)
        doc_id = change.document_id
        versions = await self._docs.list_versions(change.tenant_id, doc_id)
        tombstone = _tombstone(versions)
        live = [v for v in versions if v.status is not S.DELETED]
        if not live:
            unknown = not versions
            if tombstone is not None and change.occurred_at <= tombstone:
                return ChangeOutcome(ChangeAction.IGNORED_STALE, doc_id)
            action = ChangeAction.IGNORED_UNKNOWN if unknown else ChangeAction.IGNORED_STALE
            return ChangeOutcome(action, doc_id)
        current = max(live, key=lambda v: v.version)
        if change.occurred_at < current.acl_effective_at:
            return ChangeOutcome(ChangeAction.IGNORED_STALE, doc_id, current.version)
        await self._apply_acl(change, new_acl)
        return ChangeOutcome(ChangeAction.ACL_APPLIED, doc_id, current.version)

    async def _apply_acl(self, change: SourceChange, acl: DocumentAcl) -> None:
        await self._docs.update_acl(
            change.tenant_id, change.document_id, acl, as_of=change.occurred_at
        )
        await self._index.update_acl(change.tenant_id, change.document_id, acl)

    # ---- DELETE -----------------------------------------------------------------------------

    async def _delete(self, change: SourceChange) -> ChangeOutcome:
        doc_id = change.document_id
        versions = await self._docs.list_versions(change.tenant_id, doc_id)
        if not versions:
            return ChangeOutcome(ChangeAction.IGNORED_UNKNOWN, doc_id)
        live = [v for v in versions if v.status is not S.DELETED]
        if live and change.occurred_at < max(v.last_modified for v in live):
            return ChangeOutcome(ChangeAction.IGNORED_STALE, doc_id)  # re-created after this delete
        moved = await self._docs.mark_deleted(change.tenant_id, doc_id, change.occurred_at)
        await self._index.delete_document(change.tenant_id, doc_id)  # also on duplicates: heals
        action = ChangeAction.DELETED if moved else ChangeAction.DUPLICATE
        return ChangeOutcome(action, doc_id)


def _tombstone(versions: list[Document]) -> datetime | None:
    stamps = [v.deleted_at for v in versions if v.deleted_at is not None]
    return max(stamps) if stamps else None


def _required[T](value: T | None) -> T:
    if value is None:  # the SourceChange contract guarantees this; fail loudly if it ever lies
        raise ChangeRejected("malformed_event", retryable=False)
    return value
