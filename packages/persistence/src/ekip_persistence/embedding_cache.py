"""Redis-backed embedding cache (implements ekip_llm_gateway's EmbeddingCache by shape).

The cache is an OPTIMISATION: if Redis is down or returns junk, the caller simply re-embeds. It
never raises into the ingestion path. Vectors are stored as float32 (half the memory of float64;
the index stores float32 as well).
"""

import array
import sys
from collections.abc import Mapping, Sequence

from redis.asyncio import Redis
from redis.exceptions import RedisError

Vector = tuple[float, ...]
DEFAULT_TTL_S = 30 * 24 * 3600


def _pack(vector: Vector) -> bytes:
    a = array.array("f", vector)
    if sys.byteorder != "little":  # pragma: no cover - keep the wire format fixed
        a.byteswap()
    return a.tobytes()


def _unpack(blob: bytes) -> Vector | None:
    if not blob or len(blob) % 4:
        return None
    a = array.array("f")
    a.frombytes(blob)
    if sys.byteorder != "little":  # pragma: no cover
        a.byteswap()
    return tuple(a)


class RedisEmbeddingCache:
    def __init__(self, redis: Redis, *, ttl_s: int = DEFAULT_TTL_S, prefix: str = "ekip:") -> None:
        if ttl_s < 1:
            raise ValueError("ttl_s must be >= 1")
        self._redis = redis
        self._ttl = ttl_s
        self._prefix = prefix
        self.errors = 0  # Redis failures swallowed so far (export as a metric later)

    async def get_many(self, keys: Sequence[str]) -> dict[str, Vector]:
        if not keys:
            return {}
        try:
            blobs = await self._redis.mget([self._prefix + k for k in keys])
        except RedisError:
            self.errors += 1
            return {}
        found: dict[str, Vector] = {}
        for key, blob in zip(keys, blobs, strict=True):
            if isinstance(blob, bytes):
                vector = _unpack(blob)
                if vector is not None:
                    found[key] = vector
        return found

    async def set_many(self, items: Mapping[str, Vector]) -> None:
        if not items:
            return
        try:
            async with self._redis.pipeline(transaction=False) as pipe:
                for key, vector in items.items():
                    pipe.set(self._prefix + key, _pack(vector), ex=self._ttl)
                await pipe.execute()
        except RedisError:
            self.errors += 1
