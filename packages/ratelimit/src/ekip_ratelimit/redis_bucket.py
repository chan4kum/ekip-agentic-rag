"""Distributed token bucket in Redis: one atomic Lua script, Redis's clock."""

import math
from typing import Any

from ekip_contracts.ratelimit import RateLimitDecision, RateLimitUnavailableError
from ekip_observability.logging import get_logger
from redis.asyncio import Redis
from redis.exceptions import RedisError

log = get_logger("ekip.ratelimit")

# KEYS[1] = bucket key. ARGV = capacity, refill tokens per millisecond, cost.
# Runs atomically inside Redis, so two API replicas can never both take the last token.
# Time comes from Redis (TIME), not from the callers, so replica clock skew cannot matter.
_LUA = """
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  ts = now
end
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now)
-- an idle bucket is full again after capacity/rate ms; keep it twice that long, then forget it
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / rate) * 2)
local retry_ms = 0
if allowed == 0 then
  retry_ms = math.ceil((cost - tokens) / rate)
end
return {allowed, math.floor(tokens), retry_ms}
"""


class RedisRateLimiter:
    def __init__(
        self,
        redis: Redis,
        *,
        capacity: int,
        refill_per_second: float,
        prefix: str = "ekip:rl",
    ) -> None:
        if capacity < 1 or refill_per_second <= 0:
            raise ValueError("capacity must be >= 1 and refill_per_second > 0")
        self._redis = redis
        self._capacity = capacity
        self._rate_per_ms = refill_per_second / 1000.0
        self._prefix = prefix
        self._script: Any = redis.register_script(_LUA)

    async def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
        try:
            allowed, remaining, retry_ms = await self._script(
                keys=[f"{self._prefix}:{key}"],
                args=[self._capacity, self._rate_per_ms, cost],
            )
        except (RedisError, OSError) as exc:
            log.error("rate_limiter_unavailable", error_type=type(exc).__name__)
            raise RateLimitUnavailableError from exc
        return RateLimitDecision(
            allowed=bool(allowed),
            limit=self._capacity,
            remaining=int(remaining),
            retry_after_s=math.ceil(int(retry_ms)) / 1000.0,
        )
