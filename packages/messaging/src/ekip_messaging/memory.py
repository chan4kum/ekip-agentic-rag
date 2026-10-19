"""In-memory broker with Kafka's delivery semantics, for tests and local runs.

One partition per topic (global order per topic), consumer groups with a committed offset and an
in-flight position. `restart()` throws away the position, exactly like a crashed consumer that
comes back and resumes from its last COMMITTED offset, so uncommitted messages are redelivered.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from ekip_contracts.messaging import Delivery, MessagingError


class InMemoryBroker:
    def __init__(self) -> None:
        self._log: dict[str, list[Delivery]] = defaultdict(list)
        self._committed: dict[tuple[str, str], int] = defaultdict(int)  # (group, topic) -> offset
        self.fail_publish_to: set[str] = set()  # topics that currently refuse writes

    def messages(self, topic: str) -> list[Delivery]:
        return list(self._log[topic])

    def publisher(self) -> "InMemoryPublisher":
        return InMemoryPublisher(self)

    def subscription(self, group: str, topic: str) -> "InMemorySubscription":
        return InMemorySubscription(self, group, topic)


class InMemoryPublisher:
    def __init__(self, broker: InMemoryBroker) -> None:
        self._broker = broker

    async def publish(
        self, topic: str, *, key: str, value: bytes, headers: Mapping[str, str] | None = None
    ) -> None:
        if topic in self._broker.fail_publish_to:
            raise MessagingError("broker unavailable")
        log = self._broker._log[topic]  # noqa: SLF001
        log.append(Delivery(topic, 0, len(log), key, value, dict(headers or {})))

    async def close(self) -> None:
        return None


class InMemorySubscription:
    def __init__(self, broker: InMemoryBroker, group: str, topic: str) -> None:
        self._broker = broker
        self._group = group
        self._topic = topic
        self._position = broker._committed[(group, topic)]  # noqa: SLF001
        self.fail_commit = False

    async def poll(self, *, max_messages: int, timeout_s: float) -> list[Delivery]:
        log = self._broker._log[self._topic]  # noqa: SLF001
        batch = log[self._position : self._position + max_messages]
        self._position += len(batch)
        return batch

    async def commit(self, deliveries: Sequence[Delivery]) -> None:
        if self.fail_commit:
            raise MessagingError("commit failed")
        if deliveries:
            key = (self._group, self._topic)
            top = max(d.offset for d in deliveries) + 1
            self._broker._committed[key] = max(self._broker._committed[key], top)  # noqa: SLF001

    def restart(self) -> None:
        """Simulate a crash and restart: resume from the last committed offset."""
        self._position = self._broker._committed[(self._group, self._topic)]  # noqa: SLF001

    async def close(self) -> None:
        return None
