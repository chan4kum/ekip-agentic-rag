"""Port for a document SOURCE (SharePoint, Confluence, S3...), as reconciliation needs it.

Webhooks and change feeds are not guaranteed to deliver every change. Reconciliation periodically
asks the source "what exists right now?" and compares that with what we hold. The connector's only
jobs are to LIST the inventory and to FETCH bytes; deciding what changed is not its business.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ekip_contracts.documents import DocumentAcl


class SourceError(Exception):
    """The source could not be read right now (network, auth, throttling). Retry later."""


class SourceItemGoneError(SourceError):
    """The item was listed but is gone by the time it was fetched (deleted in between)."""


@dataclass(frozen=True)
class SourceItem:
    """Metadata for one document as the source describes it right now."""

    source_document_id: str
    source_version: str  # etag / revision id: changes whenever the content changes
    last_modified: datetime
    acl: DocumentAcl
    mime_type: str
    title: str
    source_uri: str
    acl_as_of: datetime | None = None  # when `acl` was read; defaults to last_modified

    @property
    def acl_effective_at(self) -> datetime:
        return self.acl_as_of or self.last_modified


class SourceConnector(Protocol):
    @property
    def source(self) -> str:
        """Stable source name, e.g. "sharepoint" (part of every document id)."""
        ...

    def inventory(self, tenant_id: str) -> AsyncIterator[SourceItem]:
        """Every document that exists now. MUST be complete: raise SourceError if the listing
        cannot be finished, never return a partial listing (absence means "deleted")."""
        ...

    async def fetch(self, tenant_id: str, source_document_id: str) -> bytes:
        """Raises SourceItemGoneError or SourceError."""
        ...
