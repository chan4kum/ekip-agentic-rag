"""In-memory ChunkIndex for tests and local development (same rules as the OpenSearch adapter)."""

from collections.abc import Sequence
from dataclasses import dataclass

from ekip_contracts.documents import ContractError, Document, DocumentAcl
from ekip_contracts.search_index import IndexEntry, StaleIndexVersionError


@dataclass
class _Stored:
    entry: IndexEntry
    acl: DocumentAcl


class InMemoryChunkIndex:
    def __init__(self, *, model_version: str, dimensions: int) -> None:
        self._model_version = model_version
        self._dims = dimensions
        self._rows: dict[tuple[str, str], _Stored] = {}  # (tenant_id, chunk_id) -> row
        self.fail_with: Exception | None = None  # set to simulate an outage on the next write
        self.write_calls = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    async def ensure_ready(self) -> None:
        return None

    async def replace_document(self, document: Document, entries: Sequence[IndexEntry]) -> int:
        self.write_calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        seen: set[str] = set()
        for e in entries:
            c = e.chunk
            if (
                c.tenant_id != document.tenant_id
                or c.document_id != document.document_id
                or c.document_version != document.version
                or e.embedding.model_version != self._model_version
                or e.embedding.dimensions != self._dims
                or c.chunk_id in seen
            ):
                raise ContractError("invalid index entry")
            seen.add(c.chunk_id)
        newest = max(
            (
                r.entry.chunk.document_version
                for r in self._document_rows(document.tenant_id, document.document_id)
            ),
            default=None,
        )
        if newest is not None and newest > document.version:
            raise StaleIndexVersionError("a newer version of this document is already indexed")
        for key in [
            (r.entry.chunk.tenant_id, r.entry.chunk.chunk_id)
            for r in self._document_rows(document.tenant_id, document.document_id)
        ]:
            del self._rows[key]
        for e in entries:
            self._rows[(e.chunk.tenant_id, e.chunk.chunk_id)] = _Stored(e, document.acl)
        return len(entries)

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        for r in self._document_rows(tenant_id, document_id):
            del self._rows[(tenant_id, r.entry.chunk.chunk_id)]

    async def update_acl(self, tenant_id: str, document_id: str, acl: DocumentAcl) -> int:
        rows = self._document_rows(tenant_id, document_id)
        for r in rows:
            r.acl = acl
        return len(rows)

    async def count_chunks(self, tenant_id: str, document_id: str) -> int:
        return len(self._document_rows(tenant_id, document_id))

    # ---- test helpers -----------------------------------------------------------------------

    def chunk_ids(self, tenant_id: str, document_id: str) -> list[str]:
        return sorted(r.entry.chunk.chunk_id for r in self._document_rows(tenant_id, document_id))

    def visible_chunk_ids(self, tenant_id: str, user: str, groups: frozenset[str]) -> list[str]:
        return sorted(
            cid
            for (t, cid), r in self._rows.items()
            if t == tenant_id and r.acl.permits(user, groups)
        )

    def _document_rows(self, tenant_id: str, document_id: str) -> list[_Stored]:
        return [
            r
            for (t, _), r in self._rows.items()
            if t == tenant_id and r.entry.chunk.document_id == document_id
        ]
