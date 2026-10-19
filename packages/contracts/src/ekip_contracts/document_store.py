"""Port for persisting document versions and their chunks.

Every method takes `tenant_id` explicitly and every implementation must put it in every query
(defence in depth on top of the token-derived tenant, invariant I-4). A row from another tenant is
indistinguishable from a missing row.
"""

from collections.abc import Sequence
from typing import Protocol

from ekip_contracts.assets import ImageAsset
from ekip_contracts.documents import Chunk, Document, DocumentStatus


class RepositoryError(Exception):
    """Base class for expected persistence failures."""


class DocumentNotFoundError(RepositoryError):
    """No such document version for this tenant (also what a cross-tenant lookup looks like)."""


class VersionConflictError(RepositoryError):
    """That (document, version) already exists with different content or source version."""


class InvalidTransitionError(RepositoryError):
    """The status change is not a legal move from the current status."""

    def __init__(self, current: DocumentStatus, target: DocumentStatus) -> None:
        super().__init__(f"cannot move from {current.value} to {target.value}")
        self.current = current
        self.target = target


class ImmutableVersionError(RepositoryError):
    """Chunks of an INDEXED or DELETED version are frozen."""


class DocumentRepository(Protocol):
    async def register(self, document: Document) -> Document:
        """Record a new PENDING version. Idempotent: registering the same (document, version)
        with the same content_hash and source_version returns the stored row unchanged; a
        different one raises VersionConflictError. The database decides, so concurrent workers
        cannot both win."""
        ...

    async def get(self, tenant_id: str, document_id: str, version: int) -> Document | None: ...

    async def latest_version(self, tenant_id: str, document_id: str) -> int | None: ...

    async def transition(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        target: DocumentStatus,
        *,
        failure_reason: str | None = None,
    ) -> Document:
        """Atomic compare-and-set: moves the status only if it is a legal step from the CURRENT
        status. Raises DocumentNotFoundError or InvalidTransitionError. `failure_reason` is stored
        only for FAILED."""
        ...

    async def replace_chunks(
        self, tenant_id: str, document_id: str, version: int, chunks: Sequence[Chunk]
    ) -> None:
        """Atomically replace the whole chunk set of one version (a retry after a crash leaves no
        partial set). Raises ImmutableVersionError once the version is INDEXED or DELETED."""
        ...

    async def get_chunks(self, tenant_id: str, document_id: str, version: int) -> list[Chunk]:
        """Chunks ordered by ordinal."""
        ...


class AssetRepository(Protocol):
    """Metadata for images extracted from a document version (bytes live in the ObjectStore)."""

    async def replace_assets(
        self, tenant_id: str, document_id: str, version: int, assets: Sequence[ImageAsset]
    ) -> None:
        """Atomically replace the image set of one version. Same freezing rule as chunks."""
        ...

    async def list_assets(self, tenant_id: str, document_id: str, version: int) -> list[ImageAsset]:
        """Ordered by ordinal."""
        ...

    async def get_asset(
        self, tenant_id: str, document_id: str, version: int, sha256: str
    ) -> ImageAsset | None:
        """Look up by content hash within one document version (a chunk's `asset_sha256`)."""
        ...
