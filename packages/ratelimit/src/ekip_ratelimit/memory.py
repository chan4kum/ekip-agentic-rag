"""In-process token bucket. Reference semantics for the Redis version; fine for tests and a
single process. NOT shared between processes, so never use it for multi-replica enforcement."""

import asyncio
import math
import time
from collections.abc import Callable

from ekip_contracts.ratelimit import RateLimitDecision


class InMemoryRateLimiter:
    def __init__(
        self,
        *,
        capacity: int,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1 or refill_per_second <= 0:
            raise ValueError("capacity must be >= 1 and refill_per_second > 0")
        self._capacity = capacity
        self._rate = refill_per_second
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)
        self._lock = asyncio.Lock()

    async def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
        async with self._lock:
            now = self._clock()
            tokens, last = self._buckets.get(key, (float(self._capacity), now))
            tokens = min(float(self._capacity), tokens + max(0.0, now - last) * self._rate)
            allowed = tokens >= cost
            if allowed:
                tokens -= cost
            self._buckets[key] = (tokens, now)
            retry = 0.0 if allowed else (cost - tokens) / self._rate
            return RateLimitDecision(
                allowed=allowed,
                limit=self._capacity,
                remaining=math.floor(tokens),
                retry_after_s=retry,
            )
