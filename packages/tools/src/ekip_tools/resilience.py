"""Failure containment for tool backends: circuit breaker, bulkhead, bounded retry.

All three are per tool, so one broken integration (a ticketing outage) cannot take down the
others or exhaust the worker. Clocks and sleeps are injected so tests need no real waiting.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum

Clock = Callable[[], float]


class ToolBackendError(Exception):
    """A backend failure. `retryable` says whether trying again can help (never for writes);
    `safe_message` is the only text that may reach the model or a log line."""

    def __init__(self, safe_message: str, *, retryable: bool = False) -> None:
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.retryable = retryable


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures; after `recovery_s` lets ONE probe
    through (half-open): success closes it, failure re-opens it."""

    def __init__(
        self, *, failure_threshold: int = 5, recovery_s: float = 30.0, clock: Clock = time.monotonic
    ) -> None:
        self._threshold = failure_threshold
        self._recovery_s = recovery_s
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._state = BreakerState.CLOSED
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        return self._state

    def allow(self) -> bool:
        if self._state is BreakerState.CLOSED:
            return True
        if self._state is BreakerState.OPEN:
            if self._clock() - self._opened_at < self._recovery_s:
                return False
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._probe_in_flight = False
        self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self._probe_in_flight = False
        self._failures += 1
        if self._state is BreakerState.HALF_OPEN or self._failures >= self._threshold:
            self._state = BreakerState.OPEN
            self._opened_at = self._clock()


class Bulkhead:
    """Caps concurrent calls to one backend. Rejects instead of queueing: queued calls under an
    outage just pile up latency (backpressure belongs at the edge)."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._in_flight = 0

    def try_acquire(self) -> bool:
        # asyncio is single-threaded and there is no await here, so check-and-increment is atomic.
        if self._in_flight >= self._limit:
            return False
        self._in_flight += 1
        return True

    def release(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)


async def retry_idempotent[T](
    operation: Callable[[], Awaitable[T]],
    *,
    retries: int,
    base_delay_s: float = 0.2,
    max_delay_s: float = 2.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `operation`, retrying ToolBackendError(retryable=True) with exponential backoff and
    jitter. Only call this for idempotent operations. Timeouts are NOT retried here."""
    attempt = 0
    while True:
        try:
            return await operation()
        except ToolBackendError as exc:
            if not exc.retryable or attempt >= retries:
                raise
            delay = min(max_delay_s, base_delay_s * (2**attempt))
            await sleep(delay * (0.5 + random.random() / 2))  # noqa: S311 - jitter, not security
            attempt += 1
