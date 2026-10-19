"""Durable ingestion of one document version, as a Temporal workflow.

Ownership boundary (ADR-001): Temporal owns *whether and when* a version is processed (retries,
timeouts, single-flight, visibility). `IngestDocument` owns *what processing means* and the
document's status. The workflow holds no business state; its input is three ids.

- Workflow id = tenant + document + version. Starting it twice never runs it twice
  (id_conflict_policy USE_EXISTING while running; a finished one is "already started").
  That is the "one worker per document version" guarantee `IngestDocument` relies on.
- A failed run can be started again (ALLOW_DUPLICATE_FAILED_ONLY): e.g. a redelivered event.
- Failures cross the activity boundary as ApplicationError carrying only a reason CODE: Temporal
  stores failure messages in workflow history, and a raw exception message can echo document text.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy, WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError
from temporalio.service import RPCError
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from ekip_contracts.messaging import ProcessingFailure
    from ekip_ingestion import IngestionFailed

PERMANENT = "IngestionFailedPermanent"
TRANSIENT = "IngestionFailedTransient"
UNEXPECTED = "IngestionUnexpected"
ACTIVITY_NAME = "ingest_version"


@dataclass(frozen=True)
class RetryConfig:
    initial_interval_s: float = 5.0
    backoff: float = 2.0
    max_interval_s: float = 300.0
    max_attempts: int = 8


@dataclass(frozen=True)
class IngestVersionInput:
    tenant_id: str
    document_id: str
    version: int
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass(frozen=True)
class IngestVersionResult:
    chunk_count: int
    image_count: int


def workflow_id(tenant_id: str, document_id: str, version: int) -> str:
    return f"ingest:{tenant_id}:{document_id}:v{version}"


@workflow.defn(name="IngestVersion")
class IngestVersionWorkflow:
    @workflow.run
    async def run(self, inp: IngestVersionInput) -> IngestVersionResult:
        result: IngestVersionResult = await workflow.execute_activity(
            ACTIVITY_NAME,
            inp,
            result_type=IngestVersionResult,
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=inp.retry.initial_interval_s),
                backoff_coefficient=inp.retry.backoff,
                maximum_interval=timedelta(seconds=inp.retry.max_interval_s),
                maximum_attempts=inp.retry.max_attempts,
                non_retryable_error_types=[PERMANENT],
            ),
        )
        return result


class Outcome(Protocol):
    @property
    def chunk_count(self) -> int: ...
    @property
    def image_count(self) -> int: ...


class Ingester(Protocol):
    """What the activity needs from the use case (IngestDocument satisfies it)."""

    async def run(self, tenant_id: str, document_id: str, version: int) -> Outcome: ...


class IngestActivities:
    def __init__(self, ingest: Ingester) -> None:
        self._ingest = ingest

    @activity.defn(name=ACTIVITY_NAME)
    async def ingest_version(self, inp: IngestVersionInput) -> IngestVersionResult:
        try:
            result = await self._ingest.run(inp.tenant_id, inp.document_id, inp.version)
        except IngestionFailed as exc:
            kind = TRANSIENT if exc.retryable else PERMANENT
            raise ApplicationError(exc.reason, type=kind, non_retryable=not exc.retryable) from None
        except Exception:  # noqa: BLE001 - scrub: the message may contain document content
            raise ApplicationError("unexpected_error", type=UNEXPECTED) from None
        return IngestVersionResult(result.chunk_count, result.image_count)


class TemporalIngestTrigger:
    """Implements `ekip_ingestion.IngestTrigger`: starts the workflow, does not wait for it."""

    def __init__(
        self,
        client: Client,
        *,
        task_queue: str,
        retry: RetryConfig = RetryConfig(),  # noqa: B008 - frozen dataclass
    ) -> None:
        self._client = client
        self._task_queue = task_queue
        self._retry = retry

    async def start(self, tenant_id: str, document_id: str, version: int) -> None:
        try:
            await self._client.start_workflow(
                IngestVersionWorkflow.run,
                IngestVersionInput(tenant_id, document_id, version, self._retry),
                id=workflow_id(tenant_id, document_id, version),
                task_queue=self._task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
                execution_timeout=timedelta(hours=6),
            )
        except WorkflowAlreadyStartedError:
            return  # this version already completed successfully: nothing to do
        except (RPCError, OSError):
            raise ProcessingFailure("workflow_unavailable", retryable=True) from None


async def connect(address: str, namespace: str = "default", **options: object) -> Client:
    return await Client.connect(address, namespace=namespace, **options)  # type: ignore[arg-type]


def build_worker(
    client: Client, ingest: Ingester, *, task_queue: str, max_concurrent_activities: int = 10
) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[IngestVersionWorkflow],
        activities=[IngestActivities(ingest).ingest_version],
        max_concurrent_activities=max_concurrent_activities,
    )
