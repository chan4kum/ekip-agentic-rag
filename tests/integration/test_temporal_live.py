"""Live workflow checks against Temporal from Compose (RUN_INTEGRATION=1).

Each test uses its own task queue and fresh document ids, so runs never see each other.
"""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from ekip_contracts.documents import DocumentStatus
from ekip_ingestion import IngestionFailed
from ekip_workflows import (
    IngestVersionResult,
    RetryConfig,
    TemporalIngestTrigger,
    build_worker,
    connect,
    workflow_id,
)
from temporalio.client import Client, WorkflowFailureError

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs live Compose stack"),
]
ADDRESS = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
FAST = RetryConfig(initial_interval_s=0.05, backoff=1.0, max_interval_s=0.1, max_attempts=4)


@dataclass(frozen=True)
class Done:
    chunk_count: int = 7
    image_count: int = 0


class Scripted:
    """Stands in for IngestDocument: each call pops the next scripted outcome."""

    def __init__(self, *outcomes: Exception | None) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def run(self, tenant_id: str, document_id: str, version: int) -> Done:
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome:
            raise outcome
        return Done()


def transient() -> IngestionFailed:
    return IngestionFailed("index_unavailable", stage=DocumentStatus.INDEXING, retryable=True)


def permanent() -> IngestionFailed:
    return IngestionFailed("parse_error", stage=DocumentStatus.PARSING, retryable=False)


@pytest.fixture
async def client() -> Client:
    return await connect(ADDRESS)


@pytest.fixture
async def doc_id() -> str:
    return f"doc_{uuid.uuid4().hex}"


@asynccontextmanager
async def running(client: Client, ingest: Scripted) -> AsyncIterator[TemporalIngestTrigger]:
    queue = f"test-{uuid.uuid4().hex}"
    async with build_worker(client, ingest, task_queue=queue):
        yield TemporalIngestTrigger(client, task_queue=queue, retry=FAST)


async def result_of(client: Client, doc_id: str, version: int = 1) -> Any:
    handle = client.get_workflow_handle(
        workflow_id("tenant-a", doc_id, version), result_type=IngestVersionResult
    )
    return await asyncio.wait_for(handle.result(), timeout=30)


async def test_a_version_runs_to_completion(client: Client, doc_id: str) -> None:
    ingest = Scripted()
    async with running(client, ingest) as trigger:
        await trigger.start("tenant-a", doc_id, 1)
        result = await result_of(client, doc_id)
    assert ingest.calls == 1 and result.chunk_count == 7


async def test_starting_the_same_version_twice_runs_it_once(client: Client, doc_id: str) -> None:
    ingest = Scripted()
    async with running(client, ingest) as trigger:
        await asyncio.gather(*(trigger.start("tenant-a", doc_id, 1) for _ in range(5)))
        await result_of(client, doc_id)
        await trigger.start("tenant-a", doc_id, 1)  # after success: accepted, no re-run
        await asyncio.sleep(0.5)
    assert ingest.calls == 1


async def test_retryable_failures_are_retried_until_success(client: Client, doc_id: str) -> None:
    ingest = Scripted(transient(), transient())
    async with running(client, ingest) as trigger:
        await trigger.start("tenant-a", doc_id, 1)
        await result_of(client, doc_id)
    assert ingest.calls == 3


async def test_a_permanent_failure_is_not_retried_but_a_new_trigger_can_rerun(
    client: Client, doc_id: str
) -> None:
    ingest = Scripted(permanent())
    async with running(client, ingest) as trigger:
        await trigger.start("tenant-a", doc_id, 1)
        with pytest.raises(WorkflowFailureError):
            await result_of(client, doc_id)
        assert ingest.calls == 1  # exactly once: no retries
        await trigger.start("tenant-a", doc_id, 1)  # e.g. a redelivered event after a fix
        await result_of(client, doc_id)
    assert ingest.calls == 2


async def test_retries_stop_at_the_attempt_limit(client: Client, doc_id: str) -> None:
    ingest = Scripted(*[transient() for _ in range(10)])
    async with running(client, ingest) as trigger:
        await trigger.start("tenant-a", doc_id, 1)
        with pytest.raises(WorkflowFailureError):
            await result_of(client, doc_id)
    assert ingest.calls == FAST.max_attempts
