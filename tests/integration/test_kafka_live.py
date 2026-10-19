"""Live Kafka-API checks against Redpanda from Compose (RUN_INTEGRATION=1)."""

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest
from ekip_contracts.documents import DocumentAcl, sha256_hex
from ekip_contracts.events import ChangeKind, SourceChange
from ekip_contracts.messaging import ChangeTopics, MessagingError
from ekip_ingestion import (
    ChangeAction,
    ChangeConsumer,
    ChangeOutcome,
    ChangePublisher,
)
from ekip_messaging import KafkaPublisher, KafkaSubscription

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs live Compose stack"),
]
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")


def event(doc: int, minute: int) -> SourceChange:
    return SourceChange(
        event_id=f"evt-{doc}-{minute}", tenant_id="tenant-a", source="sharepoint",
        source_document_id=f"doc-{doc}", kind=ChangeKind.UPSERT,
        occurred_at=datetime(2026, 10, 19, 6, minute, tzinfo=UTC), source_version=f"v{minute}",
        content_hash=sha256_hex(f"{doc}-{minute}"), mime_type="text/html", title="T",
        source_uri="https://example.test/x", acl=DocumentAcl("acl-1", frozenset({"alice"})),
    )  # fmt: skip


class Recorder:
    def __init__(self, fail_doc: str | None = None) -> None:
        self.seen: list[str] = []
        self.fail_doc = fail_doc

    async def process(self, ch: SourceChange) -> ChangeOutcome:
        from ekip_contracts.messaging import ProcessingFailure

        if ch.source_document_id == self.fail_doc:
            raise ProcessingFailure("parse_error", retryable=False)
        self.seen.append(ch.event_id)
        return ChangeOutcome(ChangeAction.DUPLICATE, ch.document_id)


async def test_publish_consume_commit_and_dead_letter_over_real_kafka() -> None:
    run = uuid.uuid4().hex[:8]
    topics = ChangeTopics.with_prefix(f"it{run}")
    group = f"g-{run}"
    pub = KafkaPublisher(BOOTSTRAP)
    live = KafkaSubscription(BOOTSTRAP, group_id=group, topics=[topics.live])
    backfill = KafkaSubscription(BOOTSTRAP, group_id=group, topics=[topics.backfill])
    try:
        publisher = ChangePublisher(pub, topics)
        for minute in (1, 2, 3):
            for doc in (1, 2, 3):
                await publisher.publish(event(doc, minute))
        await pub.publish(topics.live, key="k", value=b"{broken", headers={"origin": "it"})

        proc = Recorder(fail_doc="doc-3")
        consumer = ChangeConsumer(
            live=live, backfill=backfill, dead_letter=pub, processor=proc,  # type: ignore[arg-type]
            topics=topics, poll_timeout_s=1.0,
        )  # fmt: skip
        handled = 0
        for _ in range(20):
            handled += await consumer.run_once()
            if handled >= 10:
                break
        assert handled == 10
        for doc in (1, 2):  # per-document order survived the broker
            assert [e for e in proc.seen if e.startswith(f"evt-{doc}-")] == [
                f"evt-{doc}-{m}" for m in (1, 2, 3)
            ]
        assert consumer.dead_lettered == 4  # 3 x doc-3 (parse_error) + 1 malformed

        dlq = KafkaSubscription(BOOTSTRAP, group_id=f"dlq-{run}", topics=[topics.dead_letter])
        try:
            parked = []
            for _ in range(10):
                parked += await dlq.poll(max_messages=20, timeout_s=1.0)
                if len(parked) >= 4:
                    break
        finally:
            await dlq.close()
        reasons = sorted(d.headers["dlq-reason"] for d in parked)
        assert reasons == ["malformed_event", "parse_error", "parse_error", "parse_error"]
        assert any(d.headers.get("origin") == "it" for d in parked)  # original headers kept
    finally:
        await live.close()
        await backfill.close()
        await pub.close()

    # offsets were committed: a fresh consumer in the same group sees nothing new
    again = KafkaSubscription(BOOTSTRAP, group_id=group, topics=[topics.live])
    try:
        assert await again.poll(max_messages=10, timeout_s=2.0) == []
    finally:
        await again.close()


async def test_unreachable_broker_is_a_messaging_error() -> None:
    pub = KafkaPublisher("127.0.0.1:1", request_timeout_ms=2000)
    try:
        with pytest.raises(MessagingError):
            await asyncio.wait_for(pub.publish("t", key="k", value=b"v"), timeout=15)
    finally:
        await pub.close()
