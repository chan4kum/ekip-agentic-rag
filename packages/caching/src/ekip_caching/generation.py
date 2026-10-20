"""Per-tenant corpus generation: the counter every cache key includes.

Bumping it makes every earlier entry of that tenant unreachable at once (the old keys simply
stop matching). Used for deletes and permission changes, where waiting for background cleanup
is not acceptable.
"""

import time
from collections.abc import Callable

from ekip_caching.keys import DEFAULT_PREFIX, generation_key
from ekip_caching.store import CacheStore


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


class CorpusGenerations:
    """A missing counter starts at the current time in ms, never at 0, so a counter lost to
    eviction or a flush cannot repeat an old value and revive old keys."""

    def __init__(
        self,
        store: CacheStore,
        *,
        prefix: str = DEFAULT_PREFIX,
        now_ms: Callable[[], int] = _now_ms,
    ) -> None:
        self._store = store
        self._prefix = prefix
        self._now_ms = now_ms

    async def current(self, tenant_id: str) -> int:
        return await self._store.get_or_init_counter(
            generation_key(tenant_id, prefix=self._prefix), self._now_ms()
        )

    async def bump(self, tenant_id: str) -> int:
        key = generation_key(tenant_id, prefix=self._prefix)
        await self._store.get_or_init_counter(key, self._now_ms())  # never INCR from nothing
        return await self._store.incr(key)
