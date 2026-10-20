"""Choose where conversation memory lives (used only by the composition root)."""

import asyncpg
from ekip_config.settings import Settings
from ekip_contracts.conversation import ConversationKey, ConversationStore, Turn
from ekip_persistence import InMemoryConversationStore, PostgresConversationStore, create_pool


class DeferredPostgresConversationStore:
    """The Postgres pool can only be opened inside the running event loop, but the app is wired
    synchronously. This holds the place until `open()` runs at startup. Used before that, it
    raises, and the agent's memory nodes treat that as 'no memory' (best effort by design)."""

    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: asyncpg.Pool | None = None
        self._inner: PostgresConversationStore | None = None

    async def open(self) -> None:
        self._pool = await create_pool(self._url, min_size=1, max_size=5)
        self._inner = PostgresConversationStore(self._pool)

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = self._inner = None

    def _store(self) -> PostgresConversationStore:
        if self._inner is None:
            raise RuntimeError("conversation store is not open")
        return self._inner

    async def append(self, key: ConversationKey, turn: Turn) -> None:
        await self._store().append(key, turn)

    async def recent(self, key: ConversationKey, limit: int = 6) -> list[Turn]:
        return await self._store().recent(key, limit)

    async def delete(self, key: ConversationKey) -> int:
        return await self._store().delete(key)


def build_conversation_store(
    settings: Settings,
) -> tuple[ConversationStore, DeferredPostgresConversationStore | None]:
    if settings.conversation_store == "postgres":
        deferred = DeferredPostgresConversationStore(str(settings.database_url))
        return deferred, deferred
    return InMemoryConversationStore(), None
