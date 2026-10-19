"""In-memory DocumentRepository: the test twin of the Postgres adapter.

Same contract, same errors. No `await` between a check and the write in any method, so each
operation is atomic under asyncio (the Postgres adapter gets the same guarantee from SQL).
"""

from collections.abc import Sequence
from dataclasses import replace

from ekip_contracts.assets import ImageAsset
from ekip_contracts.document_store import (
    DocumentNotFoundError,
    ImmutableVersionError,
    InvalidTransitionError,
    VersionConflictError,
)
from ekip_contracts.documents import (
    Chunk,
    ContractError,
    Document,
    DocumentStatus,
    can_transition,
)

from ekip_persistence.checks import check_assets, check_chunks

_Key = tuple[str, str, int]


class InMemoryDocumentRepository:
    def __init__(self) -> None:
        self._docs: dict[_Key, Document] = {}
        self._chunks: dict[_Key, list[Chunk]] = {}
        self._assets: dict[_Key, list[ImageAsset]] = {}

    async def register(self, document: Document) -> Document:
        if document.status is not DocumentStatus.PENDING:
            raise ContractError("only PENDING documents can be registered")
        key = (document.tenant_id, document.document_id, document.version)
        existing = self._docs.get(key)
        if existing is None:
            self._docs[key] = document
            return document
        if (
            existing.content_hash == document.content_hash
            and existing.source_version == document.source_version
        ):
            return existing  # idempotent re-registration
        raise VersionConflictError("version already exists with different content")

    async def get(self, tenant_id: str, document_id: str, version: int) -> Document | None:
        return self._docs.get((tenant_id, document_id, version))

    async def latest_version(self, tenant_id: str, document_id: str) -> int | None:
        versions = [v for (t, d, v) in self._docs if t == tenant_id and d == document_id]
        return max(versions) if versions else None

    async def transition(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        target: DocumentStatus,
        *,
        failure_reason: str | None = None,
    ) -> Document:
        key = (tenant_id, document_id, version)
        current = self._docs.get(key)
        if current is None:
            raise DocumentNotFoundError("document version not found")
        if not can_transition(current.status, target):
            raise InvalidTransitionError(current.status, target)
        reason = failure_reason if target is DocumentStatus.FAILED else None
        updated = replace(current, status=target, failure_reason=reason)
        self._docs[key] = updated
        return updated

    async def replace_chunks(
        self, tenant_id: str, document_id: str, version: int, chunks: Sequence[Chunk]
    ) -> None:
        key = (tenant_id, document_id, version)
        current = self._docs.get(key)
        if current is None:
            raise DocumentNotFoundError("document version not found")
        if current.status in (DocumentStatus.INDEXED, DocumentStatus.DELETED):
            raise ImmutableVersionError("chunks of an indexed or deleted version are frozen")
        check_chunks(tenant_id, document_id, version, chunks)
        self._chunks[key] = sorted(chunks, key=lambda c: c.ordinal)

    async def get_chunks(self, tenant_id: str, document_id: str, version: int) -> list[Chunk]:
        return list(self._chunks.get((tenant_id, document_id, version), []))

    async def replace_assets(
        self, tenant_id: str, document_id: str, version: int, assets: Sequence[ImageAsset]
    ) -> None:
        key = (tenant_id, document_id, version)
        current = self._docs.get(key)
        if current is None:
            raise DocumentNotFoundError("document version not found")
        if current.status in (DocumentStatus.INDEXED, DocumentStatus.DELETED):
            raise ImmutableVersionError("assets of an indexed or deleted version are frozen")
        check_assets(tenant_id, document_id, version, assets)
        self._assets[key] = sorted(assets, key=lambda a: a.ordinal)

    async def list_assets(self, tenant_id: str, document_id: str, version: int) -> list[ImageAsset]:
        return list(self._assets.get((tenant_id, document_id, version), []))

    async def get_asset(
        self, tenant_id: str, document_id: str, version: int, sha256: str
    ) -> ImageAsset | None:
        for a in self._assets.get((tenant_id, document_id, version), []):
            if a.sha256 == sha256:
                return a
        return None
