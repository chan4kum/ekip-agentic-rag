"""Where a paused or long run is saved: nowhere (the default), in memory, or in Postgres."""

import asyncio
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

CheckpointerKind = Literal["none", "memory", "postgres"]


def to_psycopg_url(url: str) -> str:
    """Our DATABASE_URL names the asyncpg driver; psycopg wants a plain libpq URL."""
    scheme, sep, rest = url.partition("://")
    return f"{scheme.split('+', 1)[0]}{sep}{rest}"


def _build_saver(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    """`AsyncPostgresSaver.__init__` reads the running event loop, which only the blocking
    (`.list()`/`.get()`) methods ever use. We call only the async ones, so when the app is wired
    outside a loop we build the saver inside a short-lived one."""

    async def make() -> AsyncPostgresSaver:
        return AsyncPostgresSaver(pool)  # type: ignore[arg-type]

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(make())
    return AsyncPostgresSaver(pool)  # type: ignore[arg-type]


class PostgresCheckpointer:
    """The pool cannot be opened until the event loop runs, but the app is wired synchronously.
    Construction is cheap and does no I/O; `open()` (startup) connects and creates the tables,
    `aclose()` (shutdown) releases the pool."""

    def __init__(self, url: str, *, max_size: int = 5) -> None:
        self._pool: AsyncConnectionPool = AsyncConnectionPool(
            conninfo=to_psycopg_url(url),
            min_size=1,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        self.saver = _build_saver(self._pool)

    async def open(self) -> None:
        await self._pool.open(wait=True, timeout=10)
        await self.saver.setup()

    async def aclose(self) -> None:
        await self._pool.close()


def create_checkpointer(
    kind: CheckpointerKind, database_url: str | None = None
) -> tuple[BaseCheckpointSaver[Any] | None, PostgresCheckpointer | None]:
    """Returns (saver to compile the graph with, lifecycle owner to open/close or None)."""
    if kind == "none":
        return None, None
    if kind == "memory":
        return InMemorySaver(), None
    if not database_url:
        raise ValueError("a postgres checkpointer needs a database url")
    owner = PostgresCheckpointer(database_url)
    return owner.saver, owner
