"""Kafka adapters (aiokafka). Work with Kafka, Redpanda and managed Kafka services.

Producer: idempotent, acks=all (a message that `publish` returned for is durable on the brokers).
Consumer: auto-commit OFF; the caller commits after it has handled the messages (at-least-once).
Errors are reported as MessagingError without broker messages (they can name topics and hosts).
Pass `client_options` for TLS / SASL in real environments (security_protocol, sasl_*, ssl_context).
"""

from collections.abc import Mapping, Sequence
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import KafkaError
from ekip_contracts.messaging import Delivery, MessagingError


class KafkaPublisher:
    def __init__(
        self, bootstrap_servers: str, *, client_id: str = "ekip-publisher", **client_options: Any
    ) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            client_id=client_id,
            acks="all",
            enable_idempotence=True,
            **client_options,
        )
        self._started = False
        self._closed = False

    async def publish(
        self, topic: str, *, key: str, value: bytes, headers: Mapping[str, str] | None = None
    ) -> None:
        try:
            if not self._started:
                await self._producer.start()
                self._started = True
            await self._producer.send_and_wait(
                topic,
                value=value,
                key=key.encode(),
                headers=[(k, v.encode()) for k, v in (headers or {}).items()],
            )
        except (KafkaError, OSError) as exc:
            raise MessagingError("could not publish to the message broker") from exc

    async def close(self) -> None:
        # Always stop, even if start() failed half way (broker unreachable): the producer owns a
        # client and background tasks from construction, so skipping stop() leaks them.
        if not self._closed:
            self._closed = True
            self._started = False
            await self._producer.stop()


class KafkaSubscription:
    def __init__(
        self,
        bootstrap_servers: str,
        *,
        group_id: str,
        topics: Sequence[str],
        client_id: str = "ekip-consumer",
        **client_options: Any,
    ) -> None:
        self._consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            client_id=client_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            isolation_level="read_committed",
            **client_options,
        )
        self._started = False

    async def poll(self, *, max_messages: int, timeout_s: float) -> list[Delivery]:
        try:
            if not self._started:
                await self._consumer.start()
                self._started = True
            batches = await self._consumer.getmany(
                timeout_ms=max(0, int(timeout_s * 1000)), max_records=max_messages
            )
        except (KafkaError, OSError) as exc:
            raise MessagingError("could not read from the message broker") from exc
        out: list[Delivery] = []
        for tp, records in batches.items():
            for r in records:
                out.append(
                    Delivery(
                        topic=tp.topic,
                        partition=tp.partition,
                        offset=r.offset,
                        key=r.key.decode() if r.key else None,
                        value=r.value or b"",
                        headers={k: v.decode(errors="replace") for k, v in (r.headers or [])},
                    )
                )
        return out

    async def commit(self, deliveries: Sequence[Delivery]) -> None:
        highest: dict[TopicPartition, int] = {}
        for d in deliveries:
            tp = TopicPartition(d.topic, d.partition)
            highest[tp] = max(highest.get(tp, -1), d.offset)
        if not highest:
            return
        try:
            await self._consumer.commit({tp: off + 1 for tp, off in highest.items()})
        except (KafkaError, OSError) as exc:
            raise MessagingError("could not commit offsets") from exc

    async def close(self) -> None:
        if self._started:
            await self._consumer.stop()
            self._started = False
