"""Versioned document and chunk contracts shared by ingestion, indexing and retrieval.

Design rules (see ADR-001/006, invariants I-1, I-4):
- Identity is derived, never trusted: `document_id` is a hash of (tenant, source, source id), so the
  same source document always maps to the same id, and two tenants can never collide.
- Immutable values (frozen dataclasses). A change is a NEW version, not an edit.
- Validation lives in the constructor, so an invalid object cannot exist anywhere downstream.
- ACL is part of the record and fail-closed: an empty ACL grants access to nobody.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

TENANT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # same rule the token validator applies
SOURCE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{32}$")
CHUNK_ID = re.compile(r"^(doc_[0-9a-f]{32}):v([1-9][0-9]{0,8}):c([0-9]{4,9})$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
MODEL_VERSION = re.compile(r"^[A-Za-z0-9._:/@-]{1,128}$")  # e.g. gemini-embedding-001@768
PRINCIPAL = re.compile(r"^[A-Za-z0-9._@:/+=-]{1,256}$")  # user ids and group names

MAX_SOURCE_ID_CHARS = 512
MAX_TITLE_CHARS = 512
MAX_CHUNK_CHARS = 20_000
MAX_SECTION_DEPTH = 12
MAX_ACL_PRINCIPALS = 10_000
MAX_FAILURE_CHARS = 1_000


class ContractError(ValueError):
    """A contract object was constructed with invalid data."""


class DocumentStatus(StrEnum):
    """Lifecycle of one document VERSION. Only INDEXED versions are ever searchable."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


# Legal moves. A worker may only advance a document along this graph, which is what stops two
# workers (or a stale retry) from jumping a version backwards or reviving a deleted one.
_ALLOWED_NEXT: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.PENDING: frozenset({DocumentStatus.PARSING}),
    DocumentStatus.PARSING: frozenset({DocumentStatus.CHUNKING}),
    DocumentStatus.CHUNKING: frozenset({DocumentStatus.EMBEDDING}),
    DocumentStatus.EMBEDDING: frozenset({DocumentStatus.INDEXING}),
    DocumentStatus.INDEXING: frozenset({DocumentStatus.INDEXED}),
    DocumentStatus.INDEXED: frozenset(),
    # a failed version may resume at any stage (e.g. retry only indexing after a good embedding)
    DocumentStatus.FAILED: frozenset(
        {
            DocumentStatus.PENDING,
            DocumentStatus.PARSING,
            DocumentStatus.CHUNKING,
            DocumentStatus.EMBEDDING,
            DocumentStatus.INDEXING,
        }
    ),
    DocumentStatus.DELETED: frozenset(),
}
_IN_FLIGHT = frozenset(
    {
        DocumentStatus.PENDING,
        DocumentStatus.PARSING,
        DocumentStatus.CHUNKING,
        DocumentStatus.EMBEDDING,
        DocumentStatus.INDEXING,
    }
)
for _s in _IN_FLIGHT:
    _ALLOWED_NEXT[_s] = _ALLOWED_NEXT[_s] | {DocumentStatus.FAILED, DocumentStatus.DELETED}
_ALLOWED_NEXT[DocumentStatus.INDEXED] = frozenset({DocumentStatus.DELETED})
_ALLOWED_NEXT[DocumentStatus.FAILED] = _ALLOWED_NEXT[DocumentStatus.FAILED] | {
    DocumentStatus.DELETED
}


def can_transition(current: DocumentStatus, target: DocumentStatus) -> bool:
    return target in _ALLOWED_NEXT[current]


def statuses_that_may_move_to(target: DocumentStatus) -> frozenset[DocumentStatus]:
    """The set of current statuses from which `target` is a legal next step."""
    return frozenset(s for s in DocumentStatus if can_transition(s, target))


class ChunkKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"  # text derived from an image asset; `asset_sha256` points at the bytes


def sha256_hex(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def document_id_for(tenant_id: str, source: str, source_document_id: str) -> str:
    """Stable, tenant-namespaced id. Length-prefixed fields so ("a","bc") != ("ab","c")."""
    _require(TENANT_ID.fullmatch(tenant_id), "invalid tenant_id")
    _require(SOURCE_NAME.fullmatch(source), "invalid source")
    _require_text(source_document_id, MAX_SOURCE_ID_CHARS, "source_document_id")
    material = "".join(f"{len(p)}:{p}|" for p in (tenant_id, source, source_document_id))
    return "doc_" + sha256_hex(material)[:32]


def chunk_id_for(document_id: str, version: int, ordinal: int) -> str:
    _require(DOCUMENT_ID.fullmatch(document_id), "invalid document_id")
    _require(version >= 1, "version must be >= 1")
    _require(ordinal >= 0, "ordinal must be >= 0")
    return f"{document_id}:v{version}:c{ordinal:04d}"


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_text(value: str, max_chars: int, name: str) -> None:
    _require(isinstance(value, str) and value.strip(), f"{name} must be non-empty text")
    _require(len(value) <= max_chars, f"{name} too long")
    _require("\x00" not in value, f"{name} contains a NUL byte")


def _require_aware(value: datetime, name: str) -> None:
    _require(value.tzinfo is not None and value.utcoffset() is not None, f"{name} must be tz-aware")


@dataclass(frozen=True)
class DocumentAcl:
    """Who may read a document. Fail-closed: both sets empty => nobody (not 'everybody')."""

    acl_version: str
    allowed_users: frozenset[str] = field(default_factory=frozenset)
    allowed_groups: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        _require(MODEL_VERSION.fullmatch(self.acl_version), "invalid acl_version")
        _require(
            len(self.allowed_users) + len(self.allowed_groups) <= MAX_ACL_PRINCIPALS,
            "too many ACL principals",
        )
        for principal in (*self.allowed_users, *self.allowed_groups):
            _require(PRINCIPAL.fullmatch(principal), "invalid ACL principal")

    @property
    def grants_anyone(self) -> bool:
        return bool(self.allowed_users or self.allowed_groups)

    def permits(self, user: str, groups: frozenset[str]) -> bool:
        return user in self.allowed_users or not self.allowed_groups.isdisjoint(groups)


@dataclass(frozen=True)
class Document:
    """One immutable VERSION of a source document (the logical document is `document_id`)."""

    tenant_id: str
    source: str
    source_document_id: str
    document_id: str
    version: int  # our monotonic ingestion sequence per document_id (starts at 1)
    source_version: str  # whatever the source calls it (etag, revision id, "v7")
    content_hash: str  # sha256 of the raw bytes as fetched
    title: str
    mime_type: str
    source_uri: str
    acl: DocumentAcl
    last_modified: datetime
    status: DocumentStatus = DocumentStatus.PENDING
    failure_reason: str | None = None  # short, non-sensitive; only meaningful when FAILED

    def __post_init__(self) -> None:
        _require(
            self.failure_reason is None or len(self.failure_reason) <= MAX_FAILURE_CHARS,
            "failure_reason too long",
        )
        _require(
            self.document_id
            == document_id_for(self.tenant_id, self.source, self.source_document_id),
            "document_id does not match tenant/source/source_document_id",
        )
        _require(self.version >= 1, "version must be >= 1")
        _require_text(self.source_version, 128, "source_version")
        _require(SHA256_HEX.fullmatch(self.content_hash), "content_hash must be sha256 hex")
        _require_text(self.title, MAX_TITLE_CHARS, "title")
        _require(re.fullmatch(r"^[a-z0-9.+-]+/[a-z0-9.+-]+$", self.mime_type), "invalid mime_type")
        _require_text(self.source_uri, 2048, "source_uri")
        _require_aware(self.last_modified, "last_modified")

    @classmethod
    def create(
        cls,
        *,
        tenant_id: str,
        source: str,
        source_document_id: str,
        version: int,
        source_version: str,
        content: bytes,
        title: str,
        mime_type: str,
        source_uri: str,
        acl: DocumentAcl,
        last_modified: datetime,
    ) -> "Document":
        """Preferred constructor: derives `document_id` and `content_hash` so callers cannot get
        them wrong."""
        return cls(
            tenant_id=tenant_id,
            source=source,
            source_document_id=source_document_id,
            document_id=document_id_for(tenant_id, source, source_document_id),
            version=version,
            source_version=source_version,
            content_hash=sha256_hex(content),
            title=title,
            mime_type=mime_type,
            source_uri=source_uri,
            acl=acl,
            last_modified=last_modified,
        )

    @property
    def is_searchable(self) -> bool:
        return self.status is DocumentStatus.INDEXED


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of one document version. Text only: vectors live in ChunkEmbedding."""

    chunk_id: str
    document_id: str
    tenant_id: str
    document_version: int
    ordinal: int
    content: str
    content_hash: str  # sha256 of `content`; equal hash => the embedding can be reused
    section_path: tuple[str, ...] = ()
    page_number: int | None = None
    kind: ChunkKind = ChunkKind.TEXT
    asset_sha256: str | None = None  # required for IMAGE chunks, forbidden for TEXT chunks

    def __post_init__(self) -> None:
        _require(TENANT_ID.fullmatch(self.tenant_id), "invalid tenant_id")
        _require(
            (self.kind is ChunkKind.IMAGE)
            == (self.asset_sha256 is not None and bool(SHA256_HEX.fullmatch(self.asset_sha256))),
            "IMAGE chunks need a valid asset_sha256; TEXT chunks must not have one",
        )
        _require(
            self.chunk_id == chunk_id_for(self.document_id, self.document_version, self.ordinal),
            "chunk_id does not match document_id/version/ordinal",
        )
        _require_text(self.content, MAX_CHUNK_CHARS, "content")
        _require(
            self.content_hash == sha256_hex(self.content), "content_hash does not match content"
        )
        _require(len(self.section_path) <= MAX_SECTION_DEPTH, "section_path too deep")
        for part in self.section_path:
            _require_text(part, 256, "section_path element")
        _require(self.page_number is None or self.page_number >= 1, "page_number must be >= 1")

    @classmethod
    def create(
        cls,
        *,
        document: Document,
        ordinal: int,
        content: str,
        section_path: tuple[str, ...] = (),
        page_number: int | None = None,
        kind: ChunkKind = ChunkKind.TEXT,
        asset_sha256: str | None = None,
    ) -> "Chunk":
        return cls(
            chunk_id=chunk_id_for(document.document_id, document.version, ordinal),
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            document_version=document.version,
            ordinal=ordinal,
            content=content,
            content_hash=sha256_hex(content),
            section_path=section_path,
            page_number=page_number,
            kind=kind,
            asset_sha256=asset_sha256,
        )


@dataclass(frozen=True)
class ChunkEmbedding:
    """A vector for one chunk, tagged with the model that produced it.

    Vectors from different `model_version`s are never comparable: a model change means re-embedding,
    and the index keeps the tag so mixed generations can be detected.
    """

    chunk_id: str
    content_hash: str
    model_version: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(CHUNK_ID.fullmatch(self.chunk_id), "invalid chunk_id")
        _require(SHA256_HEX.fullmatch(self.content_hash), "invalid content_hash")
        _require(MODEL_VERSION.fullmatch(self.model_version), "invalid model_version")
        _require(1 <= len(self.vector) <= 8192, "vector dimension out of range")
        _require(all(v == v and abs(v) != float("inf") for v in self.vector), "non-finite value")

    @property
    def dimensions(self) -> int:
        return len(self.vector)
