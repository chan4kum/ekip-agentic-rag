from typing import Any

import pytest
from ekip_persistence import RedisEmbeddingCache
from redis.exceptions import ConnectionError as RedisConnectionError


class BrokenRedis:
    async def mget(self, *_: Any) -> Any:
        raise RedisConnectionError("down")

    def pipeline(self, **_: Any) -> Any:
        raise RedisConnectionError("down")


class JunkRedis:
    async def mget(self, keys: list[str]) -> list[Any]:
        return [b"abc", b"", None, b"\x00\x00\x80?"]  # bad length, empty, missing, one float32 1.0


async def test_redis_outage_degrades_to_a_miss_and_never_raises() -> None:
    cache = RedisEmbeddingCache(BrokenRedis())  # type: ignore[arg-type]
    assert await cache.get_many(["a", "b"]) == {}
    await cache.set_many({"a": (1.0, 0.0)})
    assert cache.errors == 2


async def test_corrupt_payloads_are_misses() -> None:
    cache = RedisEmbeddingCache(JunkRedis())  # type: ignore[arg-type]
    assert await cache.get_many(["a", "b", "c", "d"]) == {"d": (1.0,)}


def test_ttl_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RedisEmbeddingCache(BrokenRedis(), ttl_s=0)  # type: ignore[arg-type]
