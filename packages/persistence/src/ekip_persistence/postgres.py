"""asyncpg adapter for DocumentRepository.

Rules: parameterised SQL only (never string-built values); `tenant_id` in every WHERE clause;
state changes are single atomic statements (compare-and-set), never read-then-write.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import asyncpg
from ekip_contracts.assets import ImageAsset
from ekip_contracts.document_store import (
    DocumentNotFoundError,
    ImmutableVersionError,
    InvalidTransitionError,
    VersionConflictError,
)
from ekip_contracts.documents import (
    Chunk,
    ChunkKind,
    ContractError,
    Document,
    DocumentAcl,
    DocumentStatus,
    statuses_that_may_move_to,
)

from ekip_persistence.checks import check_assets, check_chunks
from ekip_persistence.dsn import to_asyncpg_dsn

_DOC_COLUMNS = (
    "tenant_id, document_id, version, source, source_document_id, source_version, content_hash, "
    "title, mime_type, source_uri, acl_version, allowed_users, allowed_groups, last_modified, "
    "status, failure_reason, acl_as_of, deleted_at"
)


async def create_pool(url: str, *, min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    pool: asyncpg.Pool = await asyncpg.create_pool(
        to_asyncpg_dsn(url), min_size=min_size, max_size=max_size, command_timeout=30
    )
    return pool


def _to_document(row: asyncpg.Record) -> Document:
    return Document(
        tenant_id=row["tenant_id"],
        source=row["source"],
        source_document_id=row["source_document_id"],
        document_id=row["document_id"],
        version=row["version"],
        source_version=row["source_version"],
        content_hash=row["content_hash"],
        title=row["title"],
        mime_type=row["mime_type"],
        source_uri=row["source_uri"],
        acl=DocumentAcl(
            acl_version=row["acl_version"],
            allowed_users=frozenset(row["allowed_users"]),
            allowed_groups=frozenset(row["allowed_groups"]),
        ),
        last_modified=row["last_modified"],
        status=DocumentStatus(row["status"]),
        failure_reason=row["failure_reason"],
        acl_as_of=row["acl_as_of"],
        deleted_at=row["deleted_at"],
    )


def _to_chunk(row: asyncpg.Record) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        tenant_id=row["tenant_id"],
        document_version=row["document_version"],
        ordinal=row["ordinal"],
        content=row["content"],
        content_hash=row["content_hash"],
        section_path=tuple(row["section_path"]),
        page_number=row["page_number"],
        kind=ChunkKind(row["kind"]),
        asset_sha256=row["asset_sha256"],
    )


def _to_asset(row: asyncpg.Record) -> ImageAsset:
    bbox = row["bbox"]
    return ImageAsset(
        tenant_id=row["tenant_id"],
        document_id=row["document_id"],
        document_version=row["document_version"],
        ordinal=row["ordinal"],
        sha256=row["sha256"],
        mime_type=row["mime_type"],
        byte_size=row["byte_size"],
        width=row["width"],
        height=row["height"],
        page_number=row["page_number"],
        bbox=None if bbox is None else (bbox[0], bbox[1], bbox[2], bbox[3]),
        nearby_chunk_ordinal=row["nearby_chunk_ordinal"],
        caption=row["caption"],
        alt_text=row["alt_text"],
        ocr_text=row["ocr_text"],
        description=row["description"],
        description_model_version=row["description_model_version"],
    )


_ASSET_COLUMNS = (
    "tenant_id, document_id, document_version, ordinal, sha256, mime_type, byte_size, width, "
    "height, page_number, bbox, nearby_chunk_ordinal, caption, alt_text, ocr_text, description, "
    "description_model_version"
)


class PostgresDocumentRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def register(self, document: Document) -> Document:
        if document.status is not DocumentStatus.PENDING:
            raise ContractError("only PENDING documents can be registered")
        d = document
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO documents (tenant_id, document_id, version, source, source_document_id,
                    source_version, content_hash, title, mime_type, source_uri, acl_version,
                    allowed_users, allowed_groups, last_modified, status, acl_as_of)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,'pending',$15)
                ON CONFLICT (tenant_id, document_id, version) DO NOTHING
                RETURNING {_DOC_COLUMNS}
                """,  # noqa: S608 - column list is a module constant, values are parameters
                d.tenant_id,
                d.document_id,
                d.version,
                d.source,
                d.source_document_id,
                d.source_version,
                d.content_hash,
                d.title,
                d.mime_type,
                d.source_uri,
                d.acl.acl_version,
                sorted(d.acl.allowed_users),
                sorted(d.acl.allowed_groups),
                d.last_modified,
                d.acl_as_of,
            )
            if row is not None:
                return _to_document(row)
            existing = await self._fetch(conn, d.tenant_id, d.document_id, d.version)
        if existing is None:  # pragma: no cover - deleted between the two statements
            raise VersionConflictError("version vanished during registration")
        if existing.content_hash == d.content_hash and existing.source_version == d.source_version:
            return existing
        raise VersionConflictError("version already exists with different content")

    async def get(self, tenant_id: str, document_id: str, version: int) -> Document | None:
        async with self._pool.acquire() as conn:
            return await self._fetch(conn, tenant_id, document_id, version)

    async def latest_version(self, tenant_id: str, document_id: str) -> int | None:
        async with self._pool.acquire() as conn:
            value: int | None = await conn.fetchval(
                "SELECT max(version) FROM documents WHERE tenant_id=$1 AND document_id=$2",
                tenant_id,
                document_id,
            )
        return value

    async def list_versions(self, tenant_id: str, document_id: str) -> list[Document]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_DOC_COLUMNS} FROM documents"  # noqa: S608 - constant column list
                " WHERE tenant_id=$1 AND document_id=$2 ORDER BY version",
                tenant_id,
                document_id,
            )
        return [_to_document(r) for r in rows]

    async def list_latest(self, tenant_id: str, source: str) -> list[Document]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT DISTINCT ON (document_id) {_DOC_COLUMNS} FROM documents"  # noqa: S608
                " WHERE tenant_id=$1 AND source=$2 ORDER BY document_id, version DESC",
                tenant_id,
                source,
            )
        return [_to_document(r) for r in rows]

    async def update_acl(
        self, tenant_id: str, document_id: str, acl: DocumentAcl, *, as_of: datetime
    ) -> int:
        async with self._pool.acquire() as conn:
            result: str = await conn.execute(
                """
                UPDATE documents SET acl_version=$3, allowed_users=$4, allowed_groups=$5,
                    acl_as_of=$6, updated_at=now()
                WHERE tenant_id=$1 AND document_id=$2 AND status <> 'deleted'
                """,
                tenant_id,
                document_id,
                acl.acl_version,
                sorted(acl.allowed_users),
                sorted(acl.allowed_groups),
                as_of,
            )
        return int(result.rsplit(" ", 1)[-1])  # "UPDATE <n>"

    async def mark_deleted(self, tenant_id: str, document_id: str, deleted_at: datetime) -> int:
        async with self._pool.acquire() as conn, conn.transaction():
            result: str = await conn.execute(
                """
                UPDATE documents SET status='deleted', failure_reason=NULL, updated_at=now()
                WHERE tenant_id=$1 AND document_id=$2 AND status <> 'deleted'
                """,
                tenant_id,
                document_id,
            )
            await conn.execute(
                """
                UPDATE documents SET deleted_at = GREATEST(coalesce(deleted_at, $3), $3)
                WHERE tenant_id=$1 AND document_id=$2
                """,
                tenant_id,
                document_id,
                deleted_at,
            )
        return int(result.rsplit(" ", 1)[-1])

    async def transition(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        target: DocumentStatus,
        *,
        failure_reason: str | None = None,
    ) -> Document:
        allowed_from = sorted(s.value for s in statuses_that_may_move_to(target))
        reason = failure_reason if target is DocumentStatus.FAILED else None
        async with self._pool.acquire() as conn:
            # One statement: the status check and the update cannot be separated by another worker.
            row = await conn.fetchrow(
                f"""
                UPDATE documents SET status=$5, failure_reason=$6, updated_at=now()
                WHERE tenant_id=$1 AND document_id=$2 AND version=$3 AND status = ANY($4::text[])
                RETURNING {_DOC_COLUMNS}
                """,  # noqa: S608 - constant column list
                tenant_id,
                document_id,
                version,
                allowed_from,
                target.value,
                reason,
            )
            if row is not None:
                return _to_document(row)
            current = await self._fetch(conn, tenant_id, document_id, version)
        if current is None:
            raise DocumentNotFoundError("document version not found")
        raise InvalidTransitionError(current.status, target)

    async def replace_chunks(
        self, tenant_id: str, document_id: str, version: int, chunks: Sequence[Chunk]
    ) -> None:
        check_chunks(tenant_id, document_id, version, chunks)
        async with self._pool.acquire() as conn, conn.transaction():
            # Lock the version row: concurrent replacers queue up instead of colliding mid-insert.
            status = await conn.fetchval(
                "SELECT status FROM documents WHERE tenant_id=$1 AND document_id=$2 AND version=$3"
                " FOR UPDATE",
                tenant_id,
                document_id,
                version,
            )
            if status is None:
                raise DocumentNotFoundError("document version not found")
            if status in (DocumentStatus.INDEXED.value, DocumentStatus.DELETED.value):
                raise ImmutableVersionError("chunks of an indexed or deleted version are frozen")
            await conn.execute(
                "DELETE FROM chunks WHERE tenant_id=$1 AND document_id=$2 AND document_version=$3",
                tenant_id,
                document_id,
                version,
            )
            await conn.executemany(
                """
                INSERT INTO chunks (tenant_id, document_id, document_version, chunk_id, ordinal,
                    content, content_hash, section_path, page_number, kind, asset_sha256)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                [
                    (
                        c.tenant_id,
                        c.document_id,
                        c.document_version,
                        c.chunk_id,
                        c.ordinal,
                        c.content,
                        c.content_hash,
                        list(c.section_path),
                        c.page_number,
                        c.kind.value,
                        c.asset_sha256,
                    )
                    for c in chunks
                ],
            )

    async def get_chunks(self, tenant_id: str, document_id: str, version: int) -> list[Chunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tenant_id, document_id, document_version, chunk_id, ordinal, content,
                       content_hash, section_path, page_number, kind, asset_sha256
                FROM chunks WHERE tenant_id=$1 AND document_id=$2 AND document_version=$3
                ORDER BY ordinal
                """,
                tenant_id,
                document_id,
                version,
            )
        return [_to_chunk(r) for r in rows]

    async def replace_assets(
        self, tenant_id: str, document_id: str, version: int, assets: Sequence[ImageAsset]
    ) -> None:
        check_assets(tenant_id, document_id, version, assets)
        async with self._pool.acquire() as conn, conn.transaction():
            status = await conn.fetchval(
                "SELECT status FROM documents WHERE tenant_id=$1 AND document_id=$2 AND version=$3"
                " FOR UPDATE",
                tenant_id,
                document_id,
                version,
            )
            if status is None:
                raise DocumentNotFoundError("document version not found")
            if status in (DocumentStatus.INDEXED.value, DocumentStatus.DELETED.value):
                raise ImmutableVersionError("assets of an indexed or deleted version are frozen")
            await conn.execute(
                "DELETE FROM image_assets WHERE tenant_id=$1 AND document_id=$2"
                " AND document_version=$3",
                tenant_id,
                document_id,
                version,
            )
            await conn.executemany(
                f"INSERT INTO image_assets ({_ASSET_COLUMNS})"  # noqa: S608 - constant columns
                " VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)",
                [
                    (
                        a.tenant_id,
                        a.document_id,
                        a.document_version,
                        a.ordinal,
                        a.sha256,
                        a.mime_type,
                        a.byte_size,
                        a.width,
                        a.height,
                        a.page_number,
                        None if a.bbox is None else list(a.bbox),
                        a.nearby_chunk_ordinal,
                        a.caption,
                        a.alt_text,
                        a.ocr_text,
                        a.description,
                        a.description_model_version,
                    )
                    for a in assets
                ],
            )

    async def list_assets(self, tenant_id: str, document_id: str, version: int) -> list[ImageAsset]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_ASSET_COLUMNS} FROM image_assets"  # noqa: S608 - constant columns
                " WHERE tenant_id=$1 AND document_id=$2 AND document_version=$3 ORDER BY ordinal",
                tenant_id,
                document_id,
                version,
            )
        return [_to_asset(r) for r in rows]

    async def get_asset(
        self, tenant_id: str, document_id: str, version: int, sha256: str
    ) -> ImageAsset | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_ASSET_COLUMNS} FROM image_assets"  # noqa: S608 - constant columns
                " WHERE tenant_id=$1 AND document_id=$2 AND document_version=$3 AND sha256=$4"
                " ORDER BY ordinal LIMIT 1",
                tenant_id,
                document_id,
                version,
                sha256,
            )
        return None if row is None else _to_asset(row)

    @staticmethod
    async def _fetch(conn: Any, tenant_id: str, document_id: str, version: int) -> Document | None:
        row = await conn.fetchrow(
            f"SELECT {_DOC_COLUMNS} FROM documents"  # noqa: S608 - constant column list
            " WHERE tenant_id=$1 AND document_id=$2 AND version=$3",
            tenant_id,
            document_id,
            version,
        )
        return None if row is None else _to_document(row)
