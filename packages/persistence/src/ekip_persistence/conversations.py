"""ConversationStore adapters: an in-memory twin and the asyncpg one (same rules, same tests)."""

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime

import asyncpg
from ekip_contracts.conversation import MAX_TURNS_KEPT, ConversationKey, Turn
from ekip_contracts.documents import ContractError

MAX_RECENT = 50


def _check_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_RECENT:
        raise ContractError(f"limit must be between 1 and {MAX_RECENT}")


class InMemoryConversationStore:
    def __init__(self, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._clock = clock
        self._threads: dict[ConversationKey, list[Turn]] = defaultdict(list)

    async def append(self, key: ConversationKey, turn: Turn) -> None:
        thread = self._threads[key]
        thread.append(Turn(turn.question, turn.answer, self._clock()))
        del thread[:-MAX_TURNS_KEPT]

    async def recent(self, key: ConversationKey, limit: int = 6) -> list[Turn]:
        _check_limit(limit)
        return list(self._threads.get(key, [])[-limit:])

    async def delete(self, key: ConversationKey) -> int:
        return len(self._threads.pop(key, []))


class PostgresConversationStore:
    """Parameterised SQL only; the full key is in every WHERE clause."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, key: ConversationKey, turn: Turn) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO conversation_turns "
                "(tenant_id, user_subject, conversation_id, question, answer) "
                "VALUES ($1, $2, $3, $4, $5)",
                key.tenant_id, key.user, key.conversation_id, turn.question, turn.answer,
            )  # fmt: skip
            await conn.execute(
                "DELETE FROM conversation_turns WHERE tenant_id = $1 AND user_subject = $2 "
                "AND conversation_id = $3 AND id NOT IN ("
                "  SELECT id FROM conversation_turns WHERE tenant_id = $1 AND user_subject = $2 "
                "  AND conversation_id = $3 ORDER BY id DESC LIMIT $4)",
                key.tenant_id, key.user, key.conversation_id, MAX_TURNS_KEPT,
            )  # fmt: skip

    async def recent(self, key: ConversationKey, limit: int = 6) -> list[Turn]:
        _check_limit(limit)
        rows = await self._pool.fetch(
            "SELECT question, answer, created_at FROM conversation_turns "
            "WHERE tenant_id = $1 AND user_subject = $2 AND conversation_id = $3 "
            "ORDER BY id DESC LIMIT $4",
            key.tenant_id, key.user, key.conversation_id, limit,
        )  # fmt: skip
        return [Turn(r["question"], r["answer"], r["created_at"]) for r in reversed(rows)]

    async def delete(self, key: ConversationKey) -> int:
        status: str = await self._pool.execute(
            "DELETE FROM conversation_turns WHERE tenant_id = $1 AND user_subject = $2 "
            "AND conversation_id = $3",
            key.tenant_id, key.user, key.conversation_id,
        )  # fmt: skip
        return int(status.rsplit(" ", 1)[-1])  # "DELETE n"
