"""One behaviour spec, run against every RateLimiter implementation.

The in-memory limiter always runs; the Redis one runs with RUN_INTEGRATION=1 against the live
stack. Same assertions for both = the in-memory twin is a trustworthy stand-in in other tests.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from ekip_config.settings import Settings
from ekip_contracts.ratelimit import RateLimiter, RateLimitUnavailableError
from ekip_ratelimit.memory import InMemoryRateLimiter
from ekip_ratelimit.redis_bucket import RedisRateLimiter
from redis.asyncio import Redis

Factory = Callable[[int, float], RateLimiter]
LIVE = os.environ.get("RUN_INTEGRATION") == "1"


@pytest.fixture(
    params=[
        "memory",
        pytest.param(
            "redis",
            marks=[
                pytest.mark.integration,
                pytest.mark.skipif(not LIVE, reason="needs live Redis (RUN_INTEGRATION=1)"),
            ],
        ),
    ]
)
async def make(request: pytest.FixtureRequest) -> AsyncIterator[Factory]:
    redis: Redis | None = None
    prefix = f"test:rl:{uuid.uuid4().hex}"

    def factory(capacity: int, per_second: float) -> RateLimiter:
        nonlocal redis
        if request.param == "memory":
            return InMemoryRateLimiter(capacity=capacity, refill_per_second=per_second)
        redis = Redis.from_url(str(Settings(_env_file=None).redis_url))
        return RedisRateLimiter(
            redis, capacity=capacity, refill_per_second=per_second, prefix=prefix
        )

    yield factory
    if redis is not None:
        async for key in redis.scan_iter(f"{prefix}:*"):
            await redis.delete(key)
        await redis.aclose()


async def test_burst_up_to_capacity_then_denied(make: Factory) -> None:
    rl = make(3, 0.1)
    results = [await rl.check("k") for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]
    assert all(r.limit == 3 for r in results)


async def test_denied_reports_a_sensible_retry_after(make: Factory) -> None:
    rl = make(1, 10)  # one token per 100 ms
    await rl.check("k")
    denied = await rl.check("k")
    assert not denied.allowed
    assert 0 < denied.retry_after_s <= 0.11
    assert (await rl.check("other")).retry_after_s == 0


async def test_keys_are_independent(make: Factory) -> None:
    rl = make(1, 0.1)
    assert (await rl.check("a")).allowed
    assert not (await rl.check("a")).allowed
    assert (await rl.check("b")).allowed


async def test_tokens_refill_over_time(make: Factory) -> None:
    rl = make(2, 20)  # one token per 50 ms
    await rl.check("k")
    await rl.check("k")
    assert not (await rl.check("k")).allowed
    await asyncio.sleep(0.15)
    assert (await rl.check("k")).allowed


async def test_refill_never_exceeds_capacity(make: Factory) -> None:
    rl = make(2, 100)
    await rl.check("k")
    await asyncio.sleep(0.2)  # would refill 20 tokens if uncapped
    assert [r.allowed for r in [await rl.check("k") for _ in range(3)]] == [True, True, False]


async def test_cost_larger_than_remaining_is_denied_without_consuming(make: Factory) -> None:
    rl = make(3, 0.1)
    await rl.check("k")
    await rl.check("k")  # 1 token left
    assert not (await rl.check("k", cost=2)).allowed
    assert (await rl.check("k")).allowed  # the single token was still there


async def test_concurrent_callers_never_exceed_capacity(make: Factory) -> None:
    rl = make(10, 0.1)
    results = await asyncio.gather(*(rl.check("k") for _ in range(50)))
    assert sum(r.allowed for r in results) == 10


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        InMemoryRateLimiter(capacity=0, refill_per_second=1)
    with pytest.raises(ValueError):
        InMemoryRateLimiter(capacity=1, refill_per_second=0)


async def test_memory_limiter_retry_after_is_exact_with_a_fake_clock() -> None:
    now = [100.0]
    rl = InMemoryRateLimiter(capacity=1, refill_per_second=2, clock=lambda: now[0])
    await rl.check("k")
    denied = await rl.check("k")
    assert denied.retry_after_s == pytest.approx(0.5)
    now[0] += 0.5
    assert (await rl.check("k")).allowed


async def test_redis_limiter_reports_unavailable_when_redis_is_down() -> None:
    dead = Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2)
    rl = RedisRateLimiter(dead, capacity=1, refill_per_second=1)
    with pytest.raises(RateLimitUnavailableError):
        await rl.check("k")
    await dead.aclose()


@pytest.mark.integration
@pytest.mark.skipif(not LIVE, reason="needs live Redis (RUN_INTEGRATION=1)")
async def test_idle_buckets_expire_in_redis() -> None:
    redis = Redis.from_url(str(Settings(_env_file=None).redis_url))
    prefix = f"test:rl:{uuid.uuid4().hex}"
    rl = RedisRateLimiter(redis, capacity=2, refill_per_second=10, prefix=prefix)
    await rl.check("k")
    ttl_ms = await redis.pttl(f"{prefix}:k")
    await redis.delete(f"{prefix}:k")
    await redis.aclose()
    assert 0 < ttl_ms <= 400  # twice the 200 ms needed to refill: abandoned keys do not pile up
