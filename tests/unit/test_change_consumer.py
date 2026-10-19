"""Event-driven ingestion over the in-memory broker: delivery guarantees and failure handling."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from ekip_contracts.documents import DocumentAcl, DocumentStatus, sha256_hex
from ekip_contracts.events import ChangeKind, Lane, SourceChange
from ekip_contracts.messaging import ChangeTopics, MessagingError, ProcessingFailure
from ekip_contracts.object_store import ObjectKind
from ekip_contracts.search_index import SearchIndexUnavailableError
from ekip_ingestion import (
    ChangeAction,
    ChangeConsumer,
    ChangeHandler,
    ChangeOutcome,
    ChangeProcessor,
    ChangePublisher,
    IngestDocument,
    InlineIngestTrigger,
    default_registry,
)
from ekip_llm_gateway import CachingEmbedder, HashingEmbedder, InMemoryEmbeddingCache
from ekip_messaging import InMemoryBroker
from ekip_persistence import InMemoryDocumentRepository, InMemoryObjectStore
from ekip_search import InMemoryChunkIndex

T0 = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
ACL = DocumentAcl("acl-1", frozenset({"alice"}), frozenset())
TOPICS = ChangeTopics()


def page(n: int) -> bytes:
    return (
        f"<html><head><title>T</title></head><body><h1>Doc {n}</h1>"
        f"<p>Maintenance procedure number {n} describes the hydraulic inspection steps.</p>"
        f"</body></html>"
    ).encode()


def change(
    doc: int,
    minute: int,
    kind: ChangeKind = ChangeKind.UPSERT,
    lane: Lane = Lane.LIVE,
    tag: str = "",
) -> SourceChange:
    base = {
        "event_id": f"evt-{doc}-{minute}{tag}",
        "tenant_id": "tenant-a",
        "source": "sharepoint",
        "source_document_id": f"doc-{doc}",
        "kind": kind,
        "occurred_at": T0 + timedelta(minutes=minute),
        "lane": lane,
    }
    if kind is ChangeKind.UPSERT:
        body = page(doc * 100 + minute)
        base |= {
            "source_version": f"v{minute}", "content_hash": sha256_hex(body),
            "mime_type": "text/html", "title": "Manual", "source_uri": "https://example.test/x",
            "acl": ACL,
        }  # fmt: skip
    return SourceChange(**base)  # type: ignore[arg-type]


class System:
    """Everything wired with in-memory adapters."""

    def __init__(self) -> None:
        self.broker = InMemoryBroker()
        self.repo = InMemoryDocumentRepository()
        self.objects = InMemoryObjectStore()
        base = HashingEmbedder(32)
        self.index = InMemoryChunkIndex(model_version=base.model_version, dimensions=32)
        ingest = IngestDocument(
            documents=self.repo, assets=self.repo, objects=self.objects, parsers=default_registry(),
            embedder=CachingEmbedder(base, InMemoryEmbeddingCache()), index=self.index,
        )  # fmt: skip
        handler = ChangeHandler(documents=self.repo, index=self.index, objects=self.objects)
        self.processor = ChangeProcessor(handler, InlineIngestTrigger(ingest))
        self.publisher = ChangePublisher(self.broker.publisher(), TOPICS)
        self.sleeps: list[float] = []

    async def publish(self, ch: SourceChange, *, with_content: bool = True) -> None:
        if with_content and ch.kind is ChangeKind.UPSERT:
            n = int(ch.source_document_id.split("-")[1]) * 100 + int(
                (ch.occurred_at - T0).total_seconds() // 60
            )
            await self.objects.put(
                ch.tenant_id, ObjectKind.RAW_DOCUMENT, page(n), content_type="text/html"
            )
        await self.publisher.publish(ch)

    def consumer(self, *, processor: object | None = None, **kw: object) -> ChangeConsumer:
        async def fake_sleep(s: float) -> None:
            self.sleeps.append(s)

        return ChangeConsumer(
            live=self.broker.subscription("g", TOPICS.live),
            backfill=self.broker.subscription("g", TOPICS.backfill),
            dead_letter=self.broker.publisher(),
            processor=processor or self.processor,  # type: ignore[arg-type]
            topics=TOPICS, sleep=fake_sleep, jitter=lambda: 1.0, poll_timeout_s=0.0, **kw,  # type: ignore[arg-type]
        )  # fmt: skip

    async def drain(self, c: ChangeConsumer, cycles: int = 20) -> None:
        for _ in range(cycles):
            if await c.run_once() == 0:
                return

    def searchable(self) -> list[str]:
        return self.index.visible_chunk_ids("tenant-a", "alice", frozenset())


class Scripted:
    """A processor that fails on a script, to exercise retry and parking."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.seen: list[str] = []

    async def process(self, ch: SourceChange) -> ChangeOutcome:
        self.seen.append(ch.event_id)
        if self.errors:
            raise self.errors.pop(0)
        return ChangeOutcome(ChangeAction.DUPLICATE, ch.document_id)


# ---- routing ---------------------------------------------------------------------------------


async def test_deletes_and_acl_changes_always_use_the_live_lane() -> None:
    s = System()
    await s.publish(change(1, 1, lane=Lane.BACKFILL))
    await s.publish(change(1, 2, ChangeKind.DELETE, lane=Lane.BACKFILL))
    assert [m.topic for m in s.broker.messages(TOPICS.backfill)] == [TOPICS.backfill]
    assert len(s.broker.messages(TOPICS.live)) == 1  # the delete jumped the queue
    assert s.broker.messages(TOPICS.live)[0].headers["kind"] == "delete"


async def test_messages_are_keyed_per_document_and_carry_idempotency_headers() -> None:
    s = System()
    await s.publish(change(1, 1))
    await s.publish(change(2, 1))
    a, b = s.broker.messages(TOPICS.live)
    assert a.key != b.key and a.headers["idempotency-key"] and a.headers["event-id"]


async def test_live_lane_is_drained_before_backfill_is_read() -> None:
    s = System()
    for i in range(1, 4):
        await s.publish(change(i, 1, lane=Lane.BACKFILL))
    await s.publish(change(9, 1))  # live, published AFTER the backfill
    order: list[str] = []

    class Spy:
        async def process(self, ch: SourceChange) -> ChangeOutcome:
            order.append(ch.source_document_id)
            return ChangeOutcome(ChangeAction.DUPLICATE, ch.document_id)

    await s.drain(s.consumer(processor=Spy()))
    assert order[0] == "doc-9" and len(order) == 4


# ---- end to end ------------------------------------------------------------------------------


async def test_events_become_searchable_documents_and_deletes_remove_them() -> None:
    s = System()
    for i in (1, 2, 3):
        await s.publish(change(i, 1))
    c = s.consumer()
    await s.drain(c)
    assert len(s.searchable()) >= 3 and c.processed == 3
    await s.publish(change(2, 5, ChangeKind.DELETE))
    await s.drain(c)
    assert all("doc_" in x for x in s.searchable())
    assert len(await s.repo.list_versions("tenant-a", change(2, 1).document_id)) == 1
    assert (await s.repo.list_versions("tenant-a", change(2, 1).document_id))[
        0
    ].status is DocumentStatus.DELETED


async def test_redelivering_the_same_event_does_not_duplicate_anything() -> None:
    s = System()
    ev = change(1, 1)
    await s.publish(ev)
    await s.publish(ev)  # a retry from the connector / at-least-once delivery
    await s.drain(s.consumer())
    assert len(await s.repo.list_versions("tenant-a", ev.document_id)) == 1


async def test_a_crash_before_commit_redelivers_and_the_result_is_the_same() -> None:
    s = System()
    ev = change(1, 1)
    await s.publish(ev)
    sub = s.broker.subscription("g", TOPICS.live)
    c = ChangeConsumer(
        live=sub, backfill=None, dead_letter=s.broker.publisher(), processor=s.processor,
        topics=TOPICS, poll_timeout_s=0.0,
    )  # fmt: skip
    sub.fail_commit = True
    with pytest.raises(MessagingError):
        await c.run_once()  # handled, but the commit failed: "the process died here"
    sub.fail_commit = False
    sub.restart()  # a new process resumes from the last COMMITTED offset
    assert await c.run_once() == 1  # same message again
    assert len(await s.repo.list_versions("tenant-a", ev.document_id)) == 1
    assert await c.run_once() == 0  # and now it is committed


async def test_events_for_one_document_are_processed_in_order_across_concurrency() -> None:
    s = System()
    seen: list[tuple[str, str]] = []

    class Recorder:
        async def process(self, ch: SourceChange) -> ChangeOutcome:
            seen.append((ch.source_document_id, ch.event_id))
            await asyncio.sleep(0)  # let other groups interleave
            return ChangeOutcome(ChangeAction.DUPLICATE, ch.document_id)

    for minute in (1, 2, 3):
        for doc in (1, 2, 3):
            await s.publish(change(doc, minute), with_content=False)
    await s.drain(s.consumer(processor=Recorder()))
    for doc in (1, 2, 3):
        assert [e for d, e in seen if d == f"doc-{doc}"] == [f"evt-{doc}-{m}" for m in (1, 2, 3)]


# ---- failure handling -------------------------------------------------------------------------


async def test_malformed_messages_are_parked_and_do_not_block_the_stream() -> None:
    s = System()
    pub = s.broker.publisher()
    await pub.publish(TOPICS.live, key="k", value=b"{not json", headers={"origin": "test"})
    await s.publish(change(1, 1))
    c = s.consumer()
    await s.drain(c)
    [dead] = s.broker.messages(TOPICS.dead_letter)
    assert dead.value == b"{not json" and dead.headers["dlq-reason"] == "malformed_event"
    assert dead.headers["origin"] == "test" and dead.headers["dlq-source"].startswith(TOPICS.live)
    assert c.processed == 1 and s.searchable()


async def test_retryable_failures_back_off_then_succeed_without_parking() -> None:
    s = System()
    proc = Scripted(
        SearchIndexUnavailableError("x"), ProcessingFailure("index_unavailable", retryable=True)
    )
    await s.publish(change(1, 1), with_content=False)
    c = s.consumer(processor=proc)
    await s.drain(c)
    assert c.processed == 1 and c.dead_lettered == 0 and len(proc.seen) == 3
    assert s.sleeps == [0.5, 1.0]  # exponential backoff
    assert s.broker.messages(TOPICS.dead_letter) == []


async def test_retries_are_bounded_then_the_message_is_parked_with_a_reason_code() -> None:
    s = System()
    proc = Scripted(*[ProcessingFailure("index_unavailable", retryable=True)] * 10)
    await s.publish(change(1, 1), with_content=False)
    c = s.consumer(processor=proc, max_attempts=3)
    await s.drain(c)
    [dead] = s.broker.messages(TOPICS.dead_letter)
    assert dead.headers["dlq-reason"] == "index_unavailable_retries_exhausted"
    assert dead.headers["dlq-attempts"] == "3" and len(proc.seen) == 3


async def test_non_retryable_failures_are_parked_immediately() -> None:
    s = System()
    proc = Scripted(ProcessingFailure("parse_error", retryable=False))
    await s.publish(change(1, 1), with_content=False)
    c = s.consumer(processor=proc)
    await s.drain(c)
    [dead] = s.broker.messages(TOPICS.dead_letter)
    assert dead.headers["dlq-reason"] == "parse_error" and len(proc.seen) == 1


async def test_unexpected_errors_never_leak_their_message_into_the_dlq() -> None:
    s = System()
    proc = Scripted(*[RuntimeError("secret document text /private/path")] * 5)
    await s.publish(change(1, 1), with_content=False)
    await s.drain(s.consumer(processor=proc, max_attempts=2))
    [dead] = s.broker.messages(TOPICS.dead_letter)
    assert "secret" not in str(dead.headers) and dead.headers["dlq-reason"].startswith(
        "unexpected_error"
    )


async def test_if_parking_fails_nothing_is_committed_and_the_message_comes_back() -> None:
    s = System()
    proc = Scripted(ProcessingFailure("parse_error", retryable=False))
    await s.publish(change(1, 1), with_content=False)
    sub = s.broker.subscription("g", TOPICS.live)
    c = ChangeConsumer(
        live=sub, backfill=None, dead_letter=s.broker.publisher(), processor=proc,  # type: ignore[arg-type]
        topics=TOPICS, poll_timeout_s=0.0,
    )  # fmt: skip
    s.broker.fail_publish_to.add(TOPICS.dead_letter)
    with pytest.raises(MessagingError):
        await c.run_once()
    s.broker.fail_publish_to.clear()
    sub.restart()
    proc.errors = [ProcessingFailure("parse_error", retryable=False)]
    assert await c.run_once() == 1
    assert len(s.broker.messages(TOPICS.dead_letter)) == 1  # parked exactly once, not lost


async def test_run_loop_survives_broker_errors_and_stops_on_request() -> None:
    s = System()
    sub = s.broker.subscription("g", TOPICS.live)
    calls = {"n": 0}
    original = sub.poll

    async def flaky(**kw: object) -> list:  # type: ignore[type-arg]
        calls["n"] += 1
        if calls["n"] == 1:
            raise MessagingError("broker down")
        stop.set()
        return await original(**kw)  # type: ignore[arg-type]

    sub.poll = flaky  # type: ignore[method-assign]
    stop = asyncio.Event()
    c = ChangeConsumer(
        live=sub, backfill=None, dead_letter=s.broker.publisher(), processor=s.processor,
        topics=TOPICS, poll_timeout_s=0.0, sleep=lambda _: asyncio.sleep(0),
    )  # fmt: skip
    await asyncio.wait_for(c.run(stop), timeout=2)
    assert calls["n"] == 2


def test_consumer_rejects_nonsense_limits() -> None:
    s = System()
    with pytest.raises(ValueError):
        s.consumer(max_attempts=0)
