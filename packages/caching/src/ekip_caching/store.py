"""The byte store caches sit on: a small port with an in-memory and a Redis implementation.

Both implementations obey one contract (tests/contract/test_cache_store.py). Operations raise
CacheStoreError when the backend fails; the cache layers turn that into "recompute".
"""

import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Protocol


class CacheStore(Protocol):
    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, *, ttl_s: int, tags: Sequence[str] = ()) -> None:
        """Store with expiry. Each tag lets `delete_tag` find this key later."""
        ...

    async def delete(self, key: str) -> None: ...

    async def delete_tag(self, tag: str) -> int:
        """Delete every key stored under `tag`; returns how many were removed."""
        ...

    async def incr(self, key: str) -> int:
        """Atomically add 1 to a (non-expiring) counter and return the new value."""
        ...

    async def get_or_init_counter(self, key: str, initial: int) -> int:
        """Return the counter, creating it with `initial` if absent (atomic, no overwrite)."""
        ...


class MemoryCacheStore:
    """Process-local store with TTL and a bounded size (least recently used is evicted)."""

    def __init__(
        self, *, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[bytes, float, tuple[str, ...]]] = OrderedDict()
        self._tags: dict[str, set[str]] = {}
        self._counters: dict[str, int] = {}

    def _drop(self, key: str) -> None:
        entry = self._data.pop(key, None)
        if entry is None:
            return
        for tag in entry[2]:
            members = self._tags.get(tag)
            if members is not None:
                members.discard(key)
                if not members:
                    del self._tags[tag]

    async def get(self, key: str) -> bytes | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if entry[1] <= self._clock():
            self._drop(key)
            return None
        self._data.move_to_end(key)
        return entry[0]

    async def set(self, key: str, value: bytes, *, ttl_s: int, tags: Sequence[str] = ()) -> None:
        if ttl_s < 1:
            raise ValueError("ttl_s must be >= 1")
        self._drop(key)
        self._data[key] = (value, self._clock() + ttl_s, tuple(tags))
        for tag in tags:
            self._tags.setdefault(tag, set()).add(key)
        while len(self._data) > self._max:
            self._drop(next(iter(self._data)))

    async def delete(self, key: str) -> None:
        self._drop(key)

    async def delete_tag(self, tag: str) -> int:
        keys = list(self._tags.get(tag, ()))
        for key in keys:
            self._drop(key)
        return len(keys)

    async def incr(self, key: str) -> int:
        self._counters[key] = self._counters.get(key, 0) + 1
        return self._counters[key]

    async def get_or_init_counter(self, key: str, initial: int) -> int:
        return self._counters.setdefault(key, initial)
