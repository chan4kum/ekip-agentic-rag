"""Redis implementation of CacheStore.

Deployment note: run the cache on a Redis whose eviction policy cannot silently drop counters
(`volatile-lru`: only keys with a TTL are evicted; counters have none). Even if a counter were
lost, `CorpusGenerations` re-initialises it to a fresh timestamp, and every hit is re-checked by
the VisibilityChecker, so a lost counter cannot resurrect a deleted or revoked document.
"""

from collections.abc import Sequence

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ekip_caching.errors import CacheStoreError

_BACKEND_ERRORS = (RedisError, OSError, TimeoutError)


class RedisCacheStore:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self.errors = 0  # backend failures seen (also surfaced as metrics by the layers)

    def _fail(self, exc: Exception) -> CacheStoreError:
        self.errors += 1
        return CacheStoreError(type(exc).__name__)  # class name only: no hosts or credentials

    async def get(self, key: str) -> bytes | None:
        try:
            value = await self._redis.get(key)
        except _BACKEND_ERRORS as exc:
            raise self._fail(exc) from None
        return value if isinstance(value, bytes) else None

    async def set(self, key: str, value: bytes, *, ttl_s: int, tags: Sequence[str] = ()) -> None:
        if ttl_s < 1:
            raise ValueError("ttl_s must be >= 1")
        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                pipe.set(key, value, ex=ttl_s)
                for tag in tags:
                    pipe.sadd(tag, key)
                    pipe.expire(tag, ttl_s + 60, gt=True)
                await pipe.execute()
        except _BACKEND_ERRORS as exc:
            raise self._fail(exc) from None

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(key)
        except _BACKEND_ERRORS as exc:
            raise self._fail(exc) from None

    async def delete_tag(self, tag: str) -> int:
        try:
            members = await self._redis.smembers(tag)
            keys = [m for m in members if isinstance(m, bytes | str)]
            if keys:
                await self._redis.unlink(*keys)
            await self._redis.delete(tag)
        except _BACKEND_ERRORS as exc:
            raise self._fail(exc) from None
        return len(keys)

    async def incr(self, key: str) -> int:
        try:
            return int(await self._redis.incr(key))
        except _BACKEND_ERRORS as exc:
            raise self._fail(exc) from None

    async def get_or_init_counter(self, key: str, initial: int) -> int:
        try:
            await self._redis.set(key, initial, nx=True)
            return int(await self._redis.get(key) or initial)
        except (*_BACKEND_ERRORS, ValueError) as exc:
            raise self._fail(exc) from None
