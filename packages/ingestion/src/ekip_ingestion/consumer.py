"""Event-driven ingestion: publish SourceChange events, consume them safely.

Guarantees (each has a test):
- At-least-once: offsets are committed only after every message of a batch is handled OR safely
  parked in the dead-letter topic. If parking fails, nothing is committed and the batch is
  redelivered. Handlers are idempotent (ChangeHandler decides from state), so redelivery is safe.
- Per-document order: messages are keyed by tenant+document; a key's messages run one after
  another, different keys run concurrently (bounded).
- Priority: the live lane is always drained before the backfill lane is read, and deletes and
  permission changes are published to the live lane even when a backfill produced them.
- Poison messages never block the stream: malformed or non-retryable messages go to the DLQ with a
  reason CODE (never exception text); retryable failures back off and retry in place, then park.
"""

import asyncio
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol

from ekip_contracts.documents import ContractError
from ekip_contracts.events import SourceChange
from ekip_contracts.messaging import (
    ChangeTopics,
    Delivery,
    MessagePublisher,
    MessagingError,
    ProcessingFailure,
    Subscription,
)

from ekip_ingestion.changes import ChangeHandler, ChangeOutcome
from ekip_ingestion.pipeline import IngestDocument


class IngestTrigger(Protocol):
    """Starts ingestion of one registered version. Must be idempotent (Temporal workflow id)."""

    async def start(self, tenant_id: str, document_id: str, version: int) -> None: ...


class InlineIngestTrigger:
    """Runs ingestion in the consumer's own task. Simple; Gate 4.6 replaces it with Temporal."""

    def __init__(self, ingest: IngestDocument) -> None:
        self._ingest = ingest

    async def start(self, tenant_id: str, document_id: str, version: int) -> None:
        await self._ingest.run(tenant_id, document_id, version)


class ChangeProcessor:
    def __init__(self, handler: ChangeHandler, trigger: IngestTrigger) -> None:
        self._handler = handler
        self._trigger = trigger

    async def process(self, change: SourceChange) -> ChangeOutcome:
        outcome = await self._handler.handle(change)
        if outcome.needs_ingest and outcome.version is not None:
            await self._trigger.start(change.tenant_id, outcome.document_id, outcome.version)
        return outcome


class ChangePublisher:
    """Connector-side helper: puts a SourceChange on the right lane with the right key."""

    def __init__(self, publisher: MessagePublisher, topics: ChangeTopics = ChangeTopics()) -> None:  # noqa: B008
        self._publisher = publisher
        self._topics = topics

    async def publish(self, change: SourceChange) -> None:
        await self._publisher.publish(
            self._topics.topic_for(change),
            key=change.partition_key,
            value=change.to_json().encode(),
            headers={
                "event-id": change.event_id,
                "idempotency-key": change.idempotency_key,
                "kind": change.kind.value,
            },
        )


class ChangeConsumer:
    def __init__(
        self,
        *,
        live: Subscription,
        backfill: Subscription | None,
        dead_letter: MessagePublisher,
        processor: ChangeProcessor,
        topics: ChangeTopics = ChangeTopics(),  # noqa: B008
        max_attempts: int = 5,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 10.0,
        batch_size: int = 50,
        backfill_batch_size: int = 10,
        concurrency: int = 8,
        poll_timeout_s: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.5, 1.5),  # noqa: S311 - not security
    ) -> None:
        if max_attempts < 1 or concurrency < 1:
            raise ValueError("max_attempts and concurrency must be >= 1")
        self._live, self._backfill, self._dlq = live, backfill, dead_letter
        self._processor = processor
        self._topics = topics
        self._max_attempts = max_attempts
        self._base, self._cap = base_backoff_s, max_backoff_s
        self._batch, self._backfill_batch = batch_size, backfill_batch_size
        self._gate = asyncio.Semaphore(concurrency)
        self._poll_timeout = poll_timeout_s
        self._sleep, self._jitter = sleep, jitter
        self.processed = 0
        self.dead_lettered = 0

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.run_once()
            except MessagingError:  # broker hiccup: back off and try again; nothing was lost
                await self._sleep(min(self._cap, self._base * self._jitter()))

    async def run_once(self) -> int:
        """One poll-handle-commit cycle. Returns how many messages were handled."""
        sub = self._live
        batch = await self._live.poll(
            max_messages=self._batch,
            timeout_s=0.0 if self._backfill else self._poll_timeout,
        )
        if not batch and self._backfill is not None:  # live lane is idle: spend time on backfill
            sub = self._backfill
            batch = await self._backfill.poll(
                max_messages=self._backfill_batch, timeout_s=self._poll_timeout
            )
        if not batch:
            return 0
        groups: dict[str | None, list[Delivery]] = defaultdict(list)
        for d in batch:
            groups[d.key].append(d)
        await asyncio.gather(*(self._run_group(g) for g in groups.values()))
        await sub.commit(batch)  # only after EVERY message is handled or parked
        return len(batch)

    # ---- one message ------------------------------------------------------------------------

    async def _run_group(self, deliveries: list[Delivery]) -> None:
        async with self._gate:
            for d in deliveries:  # same key => strictly in order
                await self._handle(d)

    async def _handle(self, d: Delivery) -> None:
        try:
            change = SourceChange.from_json(d.value)
        except ContractError:
            await self._park(d, "malformed_event", attempts=0)
            return
        for attempt in range(1, self._max_attempts + 1):
            try:
                await self._processor.process(change)
                self.processed += 1
                return
            except ProcessingFailure as exc:
                reason, retryable = exc.reason, exc.retryable
            except Exception:  # noqa: BLE001 - classify everything; never leak the message
                reason, retryable = "unexpected_error", True
            if not retryable:
                await self._park(d, reason, attempts=attempt)
                return
            if attempt < self._max_attempts:
                await self._sleep(min(self._cap, self._base * 2 ** (attempt - 1) * self._jitter()))
        await self._park(d, f"{reason}_retries_exhausted", attempts=self._max_attempts)

    async def _park(self, d: Delivery, reason: str, *, attempts: int) -> None:
        """Move the message to the DLQ. If this fails the exception propagates, the batch is not
        committed, and the whole batch is redelivered: a message is never dropped."""
        headers = {
            **d.headers,
            "dlq-reason": reason,
            "dlq-attempts": str(attempts),
            "dlq-source": f"{d.topic}:{d.partition}:{d.offset}",
            "dlq-at": datetime.now(UTC).isoformat(),
        }
        await self._dlq.publish(
            self._topics.dead_letter, key=d.key or "unknown", value=d.value, headers=headers
        )
        self.dead_lettered += 1
