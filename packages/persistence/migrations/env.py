"""Alembic environment: async engine, URL from DATABASE_URL (no credentials in files)."""

import asyncio
import os

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config
_DEV_DEFAULT = "postgresql+asyncpg://ekip:ekip@localhost:5432/ekip"


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        if os.environ.get("APP_ENV", "development") == "production":
            raise RuntimeError("DATABASE_URL must be set when APP_ENV=production")
        url = _DEV_DEFAULT
    return url.replace("postgresql://", "postgresql+asyncpg://", 1)


def run_migrations_offline() -> None:
    context.configure(url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection):  # type: ignore[no-untyped-def]
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
