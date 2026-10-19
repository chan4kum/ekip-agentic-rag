import base64
import json

from ekip_contracts.idempotency import BeginResult, StoredResponse
from redis.asyncio import Redis


class RedisIdempotencyStore:
    """Shared across API replicas. `SET key value NX PX ttl` is the atomic 'who got there first'."""

    def __init__(self, redis: Redis, *, prefix: str = "ekip:idem") -> None:
        self._redis = redis
        self._prefix = prefix

    def _k(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    @staticmethod
    def _encode(fingerprint: str, response: StoredResponse | None) -> str:
        if response is None:
            return json.dumps({"fp": fingerprint})
        return json.dumps(
            {
                "fp": fingerprint,
                "status": response.status,
                "ct": response.content_type,
                "body": base64.b64encode(response.body).decode(),
            }
        )

    async def begin(self, key: str, fingerprint: str, *, lock_ttl_s: float) -> BeginResult:
        name = self._k(key)
        for _ in range(2):  # the second pass covers a record expiring between SET and GET
            if await self._redis.set(
                name, self._encode(fingerprint, None), nx=True, px=int(lock_ttl_s * 1000)
            ):
                return BeginResult("started")
            raw = await self._redis.get(name)
            if raw is None:
                continue
            doc = json.loads(raw)
            if doc["fp"] != fingerprint:
                return BeginResult("mismatch")
            if "status" not in doc:
                return BeginResult("in_progress")
            return BeginResult(
                "replay",
                StoredResponse(doc["status"], base64.b64decode(doc["body"]), doc["ct"]),
            )
        return BeginResult("in_progress")  # lost two races in a row: ask the client to retry

    async def complete(
        self, key: str, fingerprint: str, response: StoredResponse, *, ttl_s: float
    ) -> None:
        await self._redis.set(
            self._k(key), self._encode(fingerprint, response), px=int(ttl_s * 1000)
        )

    async def abandon(self, key: str) -> None:
        await self._redis.delete(self._k(key))
