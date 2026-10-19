"""Sample documents, chunks and vectors for index tests."""

from collections.abc import Sequence
from datetime import UTC, datetime

from ekip_contracts.documents import Chunk, ChunkEmbedding, Document, DocumentAcl
from ekip_contracts.search_index import IndexEntry

DIMS = 32
MODEL = f"hashing-bow@{DIMS}"


def acl(users: Sequence[str] = ("alice",), groups: Sequence[str] = ("eng",)) -> DocumentAcl:
    return DocumentAcl("acl-1", frozenset(users), frozenset(groups))


def document(
    *,
    tenant: str = "tenant-a",
    source_id: str = "doc-1",
    version: int = 1,
    doc_acl: DocumentAcl | None = None,
) -> Document:
    return Document.create(
        tenant_id=tenant,
        source="sharepoint",
        source_document_id=source_id,
        version=version,
        source_version=f"v{version}",
        content=f"raw-{source_id}-{version}".encode(),
        title="Landing gear manual",
        mime_type="text/plain",
        source_uri=f"https://example.test/{source_id}",
        acl=doc_acl or acl(),
        last_modified=datetime(2026, 10, 19, tzinfo=UTC),
    )


def entry(
    doc: Document, ordinal: int, text: str, vector: Sequence[float], *, model: str = MODEL
) -> IndexEntry:
    chunk = Chunk.create(document=doc, ordinal=ordinal, content=text, section_path=("Ch 3", "Gear"))
    return IndexEntry(
        chunk,
        ChunkEmbedding(chunk.chunk_id, chunk.content_hash, model, tuple(float(v) for v in vector)),
    )


def unit(axis: int, dims: int = DIMS) -> list[float]:
    v = [0.0] * dims
    v[axis % dims] = 1.0
    return v
