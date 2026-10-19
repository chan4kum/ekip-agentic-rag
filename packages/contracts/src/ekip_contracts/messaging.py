"""Ports for the event log (Kafka-compatible) and the rules for using it.

Delivery is AT LEAST ONCE: a message is committed only after it has been fully handled (or safely
parked in the dead-letter topic), so a crash re-delivers it. Handlers are idempotent, so that is
safe. Per-document order is preserved by keying every message with `SourceChange.partition_key`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ekip_contracts.events import ChangeKind, Lane, SourceChange


class MessagingError(Exception):
    """The broker could not be reached or refused the operation."""


class ProcessingFailure(Exception):
    """Base for failures that carry a machine-readable reason and a retry verdict.

    `reason` is a fixed code (never exception text: messages can echo document content).
    """

    def __init__(self, reason: str, *, retryable: bool, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason
        self.retryable = retryable


@dataclass(frozen=True)
class ChangeTopics:
    live: str = "ekip.changes.live"
    backfill: str = "ekip.changes.backfill"
    dead_letter: str = "ekip.changes.dlq"

    @classmethod
    def with_prefix(cls, prefix: str) -> "ChangeTopics":
        return cls(f"{prefix}.changes.live", f"{prefix}.changes.backfill", f"{prefix}.changes.dlq")

    def topic_for(self, change: SourceChange) -> str:
        """Revocations and deletions always take the live lane, even when a backfill produced
        them: permission removal must never queue behind a million replayed documents."""
        if change.kind is not ChangeKind.UPSERT or change.lane is Lane.LIVE:
            return self.live
        return self.backfill


@dataclass(frozen=True)
class Delivery:
    topic: str
    partition: int
    offset: int
    key: str | None
    value: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class MessagePublisher(Protocol):
    async def publish(
        self, topic: str, *, key: str, value: bytes, headers: Mapping[str, str] | None = None
    ) -> None:
        """Returns only when the broker has durably accepted the message. Raises MessagingError."""
        ...

    async def close(self) -> None: ...


class Subscription(Protocol):
    async def poll(self, *, max_messages: int, timeout_s: float) -> list[Delivery]:
        """Up to `max_messages` uncommitted messages, in order per partition."""
        ...

    async def commit(self, deliveries: Sequence[Delivery]) -> None:
        """Mark these (and everything before them in each partition) as processed."""
        ...

    async def close(self) -> None: ...
