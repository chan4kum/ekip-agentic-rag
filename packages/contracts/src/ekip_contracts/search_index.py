"""Port for the searchable chunk index (write side).

The index is a *derived* view: Postgres stays the source of truth, and the index can always be
rebuilt from it. This port covers writing and removing; reading (BM25, vector, fusion) is the
Phase 5 retrieval engine.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ekip_contracts.documents import Chunk, ChunkEmbedding, ContractError, Document, DocumentAcl


class SearchIndexError(Exception):
    """The index rejected or failed an operation permanently."""


class SearchIndexUnavailableError(SearchIndexError):
    """The index could not be reached (or kept failing) after retries; safe to retry later."""


class StaleIndexVersionError(SearchIndexError):
    """A newer version of this document is already indexed; an older one must never overwrite it."""


@dataclass(frozen=True)
class IndexEntry:
    """One chunk plus the vector that represents it."""

    chunk: Chunk
    embedding: ChunkEmbedding

    def __post_init__(self) -> None:
        if (
            self.chunk.chunk_id != self.embedding.chunk_id
            or self.chunk.content_hash != self.embedding.content_hash
        ):
            raise ContractError("embedding does not belong to this chunk (or the text changed)")


class ChunkIndex(Protocol):
    @property
    def model_version(self) -> str:
        """The vector space this index holds. Only entries of this model version are accepted."""
        ...

    async def ensure_ready(self) -> None:
        """Create the index if absent; fail loudly if an existing one has an incompatible shape."""
        ...

    async def replace_document(self, document: Document, entries: Sequence[IndexEntry]) -> int:
        """Make `entries` the complete searchable content of `document` (this version).

        Idempotent. Removes this document's chunks from older versions and any leftover chunks of
        this version. Raises StaleIndexVersionError if a newer version is already indexed.
        Returns the number of chunks now indexed for the document.
        """
        ...

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        """Remove every chunk of a document (all versions). Deleting nothing is not an error."""
        ...

    async def update_acl(self, tenant_id: str, document_id: str, acl: DocumentAcl) -> int:
        """Change who may see a document's chunks without re-embedding. Returns chunks updated."""
        ...

    async def count_chunks(self, tenant_id: str, document_id: str) -> int:
        """How many chunks are searchable for a document (used to validate a publication)."""
        ...
