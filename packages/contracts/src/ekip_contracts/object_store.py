"""Port for blob storage (raw documents, extracted images).

Keys are CONTENT-ADDRESSED and TENANT-PREFIXED, and the store computes them from the bytes:
- a caller cannot mislabel an object or invent a path (no traversal surface: keys are built only
  from a validated tenant id, a closed `ObjectKind` and a sha256);
- identical bytes dedupe within a tenant, never across tenants (no cross-tenant existence oracle).
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ekip_contracts.documents import SHA256_HEX, TENANT_ID, ContractError

_CONTENT_TYPE = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")


class ObjectKind(StrEnum):
    RAW_DOCUMENT = "raw"
    IMAGE_ASSET = "assets"


class StorageError(Exception):
    """Base class for expected storage failures."""


class ObjectNotFoundError(StorageError):
    """No such object for this tenant (also what a cross-tenant lookup looks like)."""


class ObjectTooLargeError(StorageError):
    pass


class ObjectIntegrityError(StorageError):
    """Stored bytes no longer hash to their key (corruption or tampering)."""


@dataclass(frozen=True)
class ObjectKey:
    tenant_id: str
    kind: ObjectKind
    sha256: str

    def __post_init__(self) -> None:
        if not TENANT_ID.fullmatch(self.tenant_id):
            raise ContractError("invalid tenant_id")
        if not SHA256_HEX.fullmatch(self.sha256):
            raise ContractError("invalid sha256")

    @property
    def path(self) -> str:
        """Relative, slash-separated storage path; every component is validated above."""
        return f"tenants/{self.tenant_id}/{self.kind.value}/{self.sha256[:2]}/{self.sha256}"


@dataclass(frozen=True)
class StoredObject:
    key: ObjectKey
    size: int
    content_type: str

    def __post_init__(self) -> None:
        if self.size < 0 or not _CONTENT_TYPE.fullmatch(self.content_type):
            raise ContractError("invalid stored object metadata")


def check_content_type(content_type: str) -> None:
    if not _CONTENT_TYPE.fullmatch(content_type):
        raise ContractError("invalid content_type")


class ObjectStore(Protocol):
    async def put(
        self, tenant_id: str, kind: ObjectKind, data: bytes, *, content_type: str
    ) -> StoredObject:
        """Store bytes under their sha256. Idempotent: the same bytes return the same key.
        Raises ObjectTooLargeError above the adapter's limit."""
        ...

    async def get(self, key: ObjectKey) -> bytes:
        """Raises ObjectNotFoundError, or ObjectIntegrityError if the bytes no longer match."""
        ...

    async def head(self, key: ObjectKey) -> StoredObject | None: ...

    async def delete(self, key: ObjectKey) -> bool:
        """True if something was removed."""
        ...
