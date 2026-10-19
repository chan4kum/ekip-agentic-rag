import time
from collections.abc import Callable

from ekip_contracts.idempotency import BeginResult, StoredResponse


class InMemoryIdempotencyStore:
    """Single-process store with an injectable clock (tests, local development)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        # key -> (fingerprint, response or None while in progress, expires_at)
        self._records: dict[str, tuple[str, StoredResponse | None, float]] = {}

    def _live(self, key: str) -> tuple[str, StoredResponse | None, float] | None:
        rec = self._records.get(key)
        if rec is not None and rec[2] <= self._clock():
            del self._records[key]
            return None
        return rec

    async def begin(self, key: str, fingerprint: str, *, lock_ttl_s: float) -> BeginResult:
        rec = self._live(key)  # no await between check and set: atomic within one event loop
        if rec is None:
            self._records[key] = (fingerprint, None, self._clock() + lock_ttl_s)
            return BeginResult("started")
        stored_fp, response, _ = rec
        if stored_fp != fingerprint:
            return BeginResult("mismatch")
        if response is None:
            return BeginResult("in_progress")
        return BeginResult("replay", response)

    async def complete(
        self, key: str, fingerprint: str, response: StoredResponse, *, ttl_s: float
    ) -> None:
        self._records[key] = (fingerprint, response, self._clock() + ttl_s)

    async def abandon(self, key: str) -> None:
        self._records.pop(key, None)
