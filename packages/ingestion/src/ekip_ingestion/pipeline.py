"""IngestDocument: take ONE document version from raw bytes to searchable, recoverably.

    PENDING -> PARSING -> CHUNKING -> EMBEDDING -> INDEXING -> INDEXED      (any stage -> FAILED)

Recovery model (what makes a crash or an outage safe to retry):
- Every stage persists its output before the status advances, and persistence is atomic
  (chunks are replaced as a set; the object store and the index are idempotent), so re-running a
  stage is harmless.
- `run()` always resumes from durable state: chunks already stored => skip parse and chunk.
  Image assets are written BEFORE the chunk set, so "chunks exist" implies "assets exist".
- Embeddings are not stored in Postgres; the embedding cache makes re-embedding free, so a retry
  after an indexing outage does not pay for the vectors again.
- A version found half-done (a worker died mid-stage) is moved to FAILED("interrupted") and
  resumed. This assumes ONE worker per document version; Phase 4 enforces that with a Temporal
  workflow id.
- Failure reasons stored on the document are fixed codes, never exception text (messages from
  parsers, providers or the index can echo document content).
"""

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ekip_contracts.document_store import AssetRepository, DocumentRepository, RepositoryError
from ekip_contracts.documents import (
    Chunk,
    ChunkEmbedding,
    ContractError,
    Document,
    DocumentStatus,
)
from ekip_contracts.embedding import (
    Embedder,
    EmbeddingError,
    EmbeddingTask,
    EmbeddingUnavailableError,
    ScopableEmbedder,
)
from ekip_contracts.messaging import ProcessingFailure
from ekip_contracts.object_store import ObjectKey, ObjectKind, ObjectStore, StorageError
from ekip_contracts.parsing import ParsedDocument, ParseError
from ekip_contracts.search_index import (
    ChunkIndex,
    IndexEntry,
    SearchIndexError,
    SearchIndexUnavailableError,
    StaleIndexVersionError,
)

from ekip_ingestion.chunking import ChunkingConfig, ImageEnrichment, chunk_document, embedding_text
from ekip_ingestion.parsers.registry import ParserRegistry

S = DocumentStatus


class ImageEnricher(Protocol):
    """Optional OCR / vision step. Returns text per image, keyed by the image's sha256."""

    async def enrich(self, parsed: ParsedDocument) -> Mapping[str, ImageEnrichment]: ...


class IngestionFailed(ProcessingFailure):
    """A stage failed. The document is already FAILED; `retryable` says if a retry can help."""

    def __init__(self, reason: str, *, stage: DocumentStatus, retryable: bool) -> None:
        super().__init__(
            reason, retryable=retryable, message=f"ingestion failed at {stage.value}: {reason}"
        )
        self.stage = stage


@dataclass(frozen=True)
class IngestResult:
    document: Document
    chunk_count: int
    image_count: int


class IngestDocument:
    def __init__(
        self,
        *,
        documents: DocumentRepository,
        assets: AssetRepository,
        objects: ObjectStore,
        parsers: ParserRegistry,
        embedder: Embedder,
        index: ChunkIndex,
        chunking: ChunkingConfig = ChunkingConfig(),  # noqa: B008 - frozen dataclass
        enricher: ImageEnricher | None = None,
    ) -> None:
        if embedder.model_version != index.model_version:
            raise ValueError("embedder and index must use the same model_version")
        self._docs = documents
        self._assets = assets
        self._objects = objects
        self._parsers = parsers
        self._embedder = embedder
        self._index = index
        self._chunking = chunking
        self._enricher = enricher

    async def submit(self, document: Document, content: bytes) -> Document:
        """Store the raw bytes and register the version (idempotent). Does not process it."""
        stored = await self._objects.put(
            document.tenant_id, ObjectKind.RAW_DOCUMENT, content, content_type=document.mime_type
        )
        if stored.key.sha256 != document.content_hash:
            raise ContractError("content does not match document.content_hash")
        return await self._docs.register(document)

    async def ingest(self, document: Document, content: bytes) -> IngestResult:
        await self.submit(document, content)
        return await self.run(document.tenant_id, document.document_id, document.version)

    async def run(self, tenant_id: str, document_id: str, version: int) -> IngestResult:
        doc = await self._docs.get(tenant_id, document_id, version)
        if doc is None:
            raise RepositoryError("document version not registered")
        if doc.status in (S.INDEXED, S.DELETED):
            return await self._result(doc)  # nothing to do; safe to call again
        if doc.status not in (S.PENDING, S.FAILED):
            doc = await self._docs.transition(
                tenant_id, document_id, version, S.FAILED, failure_reason="interrupted"
            )
        chunks = await self._docs.get_chunks(tenant_id, document_id, version)
        stage = S.EMBEDDING if chunks else S.PARSING
        try:
            if stage is S.PARSING:
                chunks = await self._parse_and_chunk(doc)
            return await self._embed_and_index(doc, chunks)
        except IngestionFailed:
            raise
        except Exception as exc:  # classify, record a fixed code, never the message
            raise await self._fail(doc, exc) from exc

    # ---- stages -----------------------------------------------------------------------------

    async def _parse_and_chunk(self, doc: Document) -> list[Chunk]:
        await self._advance(doc, S.PARSING)
        raw = await self._objects.get(
            ObjectKey(doc.tenant_id, ObjectKind.RAW_DOCUMENT, doc.content_hash)
        )
        parsed = await self._parsers.parse(raw, doc.mime_type)
        await self._advance(doc, S.CHUNKING)
        enrichments = await self._enricher.enrich(parsed) if self._enricher else None
        result = chunk_document(doc, parsed, self._chunking, enrichments)
        if not result.chunks:
            raise _NoContentError
        for pending in result.assets:  # bytes first, then metadata, then the chunk set (commit)
            await self._objects.put(
                doc.tenant_id,
                ObjectKind.IMAGE_ASSET,
                pending.data,
                content_type=pending.asset.mime_type,
            )
        await self._assets.replace_assets(
            doc.tenant_id, doc.document_id, doc.version, [p.asset for p in result.assets]
        )
        await self._docs.replace_chunks(doc.tenant_id, doc.document_id, doc.version, result.chunks)
        return list(result.chunks)

    async def _embed_and_index(self, doc: Document, chunks: Sequence[Chunk]) -> IngestResult:
        await self._advance(doc, S.EMBEDDING)
        embedder = (  # per-tenant cache partition when the embedder supports one
            self._embedder.scoped(doc.tenant_id)
            if isinstance(self._embedder, ScopableEmbedder)
            else self._embedder
        )
        vectors = await embedder.embed(
            [embedding_text(doc, c) for c in chunks], task=EmbeddingTask.DOCUMENT
        )
        entries = [
            IndexEntry(
                c,
                ChunkEmbedding(c.chunk_id, c.content_hash, self._embedder.model_version, v),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        await self._advance(doc, S.INDEXING)
        # Publish with the ACL as it is NOW (it may have changed while we were embedding)...
        current = await self._docs.get(doc.tenant_id, doc.document_id, doc.version)
        if current is None or current.status is S.DELETED:
            raise IngestionFailed("deleted", stage=S.INDEXING, retryable=False)
        indexed = await self._index.replace_document(current, entries)
        if indexed != len(entries):  # validate the publication before declaring success
            raise SearchIndexError("index count does not match chunk count")
        # ...and look again afterwards: a delete or ACL change that landed while we were
        # publishing updated the repository first, so the repository is the authority.
        after = await self._docs.get(doc.tenant_id, doc.document_id, doc.version)
        if after is None or after.status is S.DELETED:
            await self._index.delete_document(doc.tenant_id, doc.document_id)
            raise IngestionFailed("deleted", stage=S.INDEXING, retryable=False)
        if after.acl != current.acl:
            await self._index.update_acl(doc.tenant_id, doc.document_id, after.acl)
        done = await self._advance(doc, S.INDEXED)
        return await self._result(done)

    # ---- helpers ----------------------------------------------------------------------------

    async def _advance(self, doc: Document, target: DocumentStatus) -> Document:
        return await self._docs.transition(doc.tenant_id, doc.document_id, doc.version, target)

    async def _result(self, doc: Document) -> IngestResult:
        chunks = await self._docs.get_chunks(doc.tenant_id, doc.document_id, doc.version)
        images = await self._assets.list_assets(doc.tenant_id, doc.document_id, doc.version)
        return IngestResult(doc, len(chunks), len(images))

    async def _fail(self, doc: Document, exc: Exception) -> IngestionFailed:
        reason, retryable = _classify(exc)
        current = await self._docs.get(doc.tenant_id, doc.document_id, doc.version)
        stage = current.status if current else doc.status
        with contextlib.suppress(RepositoryError):  # e.g. deleted concurrently
            await self._docs.transition(
                doc.tenant_id, doc.document_id, doc.version, S.FAILED, failure_reason=reason
            )
        return IngestionFailed(reason, stage=stage, retryable=retryable)


class _NoContentError(Exception):
    pass


def _classify(exc: Exception) -> tuple[str, bool]:
    """(stored reason code, retryable?). Order matters: subclasses before their parents."""
    if isinstance(exc, _NoContentError):
        return "no_content", False
    if isinstance(exc, ParseError):
        return "parse_error", False
    if isinstance(exc, EmbeddingUnavailableError):
        return "embedding_unavailable", True
    if isinstance(exc, EmbeddingError):
        return "embedding_failed", False
    if isinstance(exc, StaleIndexVersionError):
        return "superseded", False
    if isinstance(exc, SearchIndexUnavailableError):
        return "index_unavailable", True
    if isinstance(exc, SearchIndexError):
        return "index_failed", False
    if isinstance(exc, StorageError):
        return "storage_failed", True
    if isinstance(exc, ContractError):
        return "invalid_document", False
    return "unexpected_error", True
