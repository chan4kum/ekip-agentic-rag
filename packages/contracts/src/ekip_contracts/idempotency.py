"""Port for idempotency records ("this exact request was already handled")."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class StoredResponse:
    status: int
    body: bytes
    content_type: str


@dataclass(frozen=True)
class BeginResult:
    """started: you own the key, go ahead. in_progress: another request holds it.
    mismatch: the key was used with a DIFFERENT request. replay: done already, here is the answer.
    """

    kind: Literal["started", "in_progress", "mismatch", "replay"]
    response: StoredResponse | None = None


class IdempotencyStore(Protocol):
    async def begin(self, key: str, fingerprint: str, *, lock_ttl_s: float) -> BeginResult: ...

    async def complete(
        self, key: str, fingerprint: str, response: StoredResponse, *, ttl_s: float
    ) -> None: ...

    async def abandon(self, key: str) -> None:
        """Release the key so the client can retry (the attempt did not produce a final result)."""
        ...
