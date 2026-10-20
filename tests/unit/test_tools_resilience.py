import pytest
from ekip_tools import Bulkhead, CircuitBreaker, ToolBackendError, retry_idempotent
from ekip_tools.resilience import BreakerState
from tools_testkit import FakeClock


def state_of(breaker: CircuitBreaker) -> BreakerState:
    return breaker.state


def test_breaker_opens_then_probes_then_closes() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=2, recovery_s=10, clock=clock)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert state_of(breaker) is BreakerState.OPEN
    assert not breaker.allow()
    clock.advance(11)
    assert breaker.allow()  # the single probe
    assert not breaker.allow()  # others wait while the probe is in flight
    breaker.record_success()
    assert state_of(breaker) is BreakerState.CLOSED
    assert breaker.allow()


def test_failed_probe_reopens_the_breaker() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_s=5, clock=clock)
    breaker.record_failure()
    clock.advance(6)
    assert breaker.allow()
    breaker.record_failure()
    assert state_of(breaker) is BreakerState.OPEN
    assert not breaker.allow()


def test_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=3, clock=FakeClock())
    for _ in range(2):
        breaker.record_failure()
    breaker.record_success()
    for _ in range(2):
        breaker.record_failure()
    assert state_of(breaker) is BreakerState.CLOSED


def test_bulkhead_rejects_beyond_its_limit_and_releases() -> None:
    bulkhead = Bulkhead(2)
    assert bulkhead.try_acquire()
    assert bulkhead.try_acquire()
    assert not bulkhead.try_acquire()
    bulkhead.release()
    assert bulkhead.try_acquire()
    for _ in range(5):
        bulkhead.release()  # extra releases never go negative
    assert bulkhead.try_acquire()


async def test_retry_only_retryable_errors_with_a_cap() -> None:
    attempts = 0
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ToolBackendError("blip", retryable=True)
        return "done"

    assert await retry_idempotent(flaky, retries=3, sleep=sleep) == "done"
    assert attempts == 3
    assert len(delays) == 2
    assert all(0 < d <= 2.0 for d in delays)


async def test_retry_stops_at_the_cap_and_never_retries_hard_errors() -> None:
    calls = 0

    async def always() -> str:
        nonlocal calls
        calls += 1
        raise ToolBackendError("down", retryable=True)

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(ToolBackendError):
        await retry_idempotent(always, retries=2, sleep=no_sleep)
    assert calls == 3

    calls = 0

    async def hard() -> str:
        nonlocal calls
        calls += 1
        raise ToolBackendError("bad request")

    with pytest.raises(ToolBackendError):
        await retry_idempotent(hard, retries=5, sleep=no_sleep)
    assert calls == 1
