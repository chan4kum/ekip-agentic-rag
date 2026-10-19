"""Run a state-changing operation at most once per (caller, Idempotency-Key)."""

import hashlib
from collections.abc import Awaitable, Callable

from ekip_contracts.idempotency import IdempotencyStore, StoredResponse
from ekip_observability.logging import get_logger

log = get_logger("ekip.api.idempotency")


class IdempotencyInProgressError(Exception):
    """Another request with the same key is still running (client should retry shortly)."""


class IdempotencyMismatchError(Exception):
    """The key was already used for a different request: a client bug we must not paper over."""


def fingerprint_request(method: str, path: str, body: bytes) -> str:
    """What makes two requests 'the same'. Length-prefixed so field boundaries cannot be forged."""
    h = hashlib.sha256()
    for part in (method.upper().encode(), path.encode(), body):
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


class IdempotencyService:
    def __init__(
        self,
        store: IdempotencyStore,
        *,
        lock_ttl_s: float = 60,
        result_ttl_s: float = 24 * 3600,
        max_body_bytes: int = 256 * 1024,
    ) -> None:
        # lock_ttl_s must exceed the request deadline (30 s), or a slow first attempt could lose
        # its lock and let a retry run the operation a second time.
        self._store = store
        self._lock_ttl_s = lock_ttl_s
        self._result_ttl_s = result_ttl_s
        self._max_body_bytes = max_body_bytes

    async def run(
        self,
        *,
        scope: str,
        key: str,
        fingerprint: str,
        produce: Callable[[], Awaitable[StoredResponse]],
    ) -> tuple[StoredResponse, bool]:
        """Return (response, replayed). `scope` = tenant + caller + route, so keys never collide
        across users, tenants or endpoints."""
        record = hashlib.sha256(f"{scope}\x00{key}".encode()).hexdigest()
        begun = await self._store.begin(record, fingerprint, lock_ttl_s=self._lock_ttl_s)
        if begun.kind == "replay" and begun.response is not None:
            log.info("idempotent_replay", scope=scope)
            return begun.response, True
        if begun.kind == "in_progress":
            raise IdempotencyInProgressError
        if begun.kind == "mismatch":
            raise IdempotencyMismatchError

        try:
            response = await produce()
        except BaseException:  # includes cancellation by the request deadline
            await self._store.abandon(record)  # nothing final happened: allow a clean retry
            raise
        if response.status >= 500 or len(response.body) > self._max_body_bytes:
            await self._store.abandon(record)  # do not freeze a server error or a huge body
        else:
            await self._store.complete(record, fingerprint, response, ttl_s=self._result_ttl_s)
        return response, False
