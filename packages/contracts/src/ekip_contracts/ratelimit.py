"""Port for rate limiting. Implementations: ekip_ratelimit (Redis, in-memory)."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class RateLimitUnavailableError(Exception):
    """The limiter could not decide (e.g. Redis down). The caller picks fail-open or closed."""


class RateLimitDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    limit: int  # bucket capacity (the burst size)
    remaining: int  # whole tokens left after this call
    retry_after_s: float  # 0 when allowed; otherwise how long until `cost` tokens exist


class RateLimiter(Protocol):
    async def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
        """Take `cost` tokens from the bucket `key`. Raise RateLimitUnavailableError if unsure."""
        ...


@dataclass(frozen=True)
class RateLimits:
    """The limiters a route applies, plus what to do when the limiter itself is down.

    `user` is keyed per authenticated caller, `tenant` per tenant (stops one noisy tenant from
    consuming capacity meant for others). fail_open=False (default) answers 503 when the limiter
    cannot decide, so spend stays bounded; True lets requests through with a warning.
    """

    user: RateLimiter
    tenant: RateLimiter
    fail_open: bool = False
