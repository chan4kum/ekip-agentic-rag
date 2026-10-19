"""Alembic migrations against a throwaway database: upgrade, downgrade, upgrade again.

Run: RUN_INTEGRATION=1 DATABASE_URL=postgresql+asyncpg://ekip:ekip@localhost:5433/ekip \
     uv run pytest tests/integration/test_migrations_live.py
"""

import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from ekip_persistence import to_asyncpg_dsn
from tests_paths import ROOT

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs live Compose stack"),
]

ADMIN_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://ekip:ekip@localhost:5432/ekip")
INI = str(ROOT / "packages/persistence/alembic.ini")


def _with_db(url: str, name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{name}"))


def alembic(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "alembic", "-c", INI, *args],
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
async def scratch_url() -> AsyncIterator[str]:
    name = f"ekip_mig_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(to_asyncpg_dsn(ADMIN_URL))
    await admin.execute(f'CREATE DATABASE "{name}"')
    try:
        yield _with_db(ADMIN_URL, name)
    finally:
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()


async def _tables(url: str) -> set[str]:
    conn = await asyncpg.connect(to_asyncpg_dsn(url))
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public'")
    finally:
        await conn.close()
    return {r["tablename"] for r in rows}


async def test_upgrade_downgrade_upgrade(scratch_url: str) -> None:
    assert (r := alembic(scratch_url, "upgrade", "head")).returncode == 0, r.stderr
    assert {"documents", "chunks", "image_assets"} <= await _tables(scratch_url)

    assert (r := alembic(scratch_url, "downgrade", "base")).returncode == 0, r.stderr
    assert not ({"documents", "chunks", "image_assets"} & await _tables(scratch_url))

    assert (r := alembic(scratch_url, "upgrade", "head")).returncode == 0, r.stderr
    assert {"documents", "chunks", "image_assets"} <= await _tables(scratch_url)


async def test_upgrade_is_idempotent_at_head(scratch_url: str) -> None:
    assert alembic(scratch_url, "upgrade", "head").returncode == 0
    assert alembic(scratch_url, "upgrade", "head").returncode == 0  # already at head: no-op


_INSERT = (
    "INSERT INTO documents (tenant_id, document_id, version, source, source_document_id,"
    " source_version, content_hash, title, mime_type, source_uri, acl_version, last_modified,"
    " status) VALUES ($1, 'doc_' || repeat('a', 32), 1, 's', 'x', 'v', repeat('a', 64), 't',"
    " 'a/b', 'u', 'acl', now(), $2)"
)


async def test_database_rejects_rows_the_contracts_would_reject(scratch_url: str) -> None:
    """Defence in depth: even a buggy writer cannot store malformed identifiers."""
    assert alembic(scratch_url, "upgrade", "head").returncode == 0
    conn = await asyncpg.connect(to_asyncpg_dsn(scratch_url))
    try:
        await conn.execute(_INSERT, "t-ok", "pending")  # the statement itself is valid
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(_INSERT, "../evil", "pending")
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(_INSERT, "t-ok2", "hacked")
    finally:
        await conn.close()
