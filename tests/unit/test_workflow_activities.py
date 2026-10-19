"""The Temporal activity and trigger, without a Temporal server.

The workflow itself (retries, single-flight) runs live: tests/integration/test_temporal_live.py.
"""

import pytest
from ekip_contracts.documents import DocumentStatus
from ekip_contracts.messaging import ProcessingFailure
from ekip_ingestion import IngestionFailed
from ekip_workflows import (
    IngestActivities,
    IngestVersionInput,
    RetryConfig,
    TemporalIngestTrigger,
    workflow_id,
)
from temporalio.exceptions import ApplicationError

INP = IngestVersionInput("tenant-a", "doc_1", 3)


class Scripted:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str, int]] = []

    async def run(self, tenant_id: str, document_id: str, version: int) -> object:
        self.calls.append((tenant_id, document_id, version))
        if self.error:
            raise self.error
        return type("R", (), {"chunk_count": 4, "image_count": 1})()


async def test_the_activity_runs_exactly_the_requested_version() -> None:
    ingest = Scripted()
    result = await IngestActivities(ingest).ingest_version(INP)  # type: ignore[arg-type]
    assert ingest.calls == [("tenant-a", "doc_1", 3)]
    assert (result.chunk_count, result.image_count) == (4, 1)


@pytest.mark.parametrize(("retryable", "non_retryable"), [(True, False), (False, True)])
async def test_failures_cross_the_boundary_as_a_code_and_the_retry_flag(
    retryable: bool, non_retryable: bool
) -> None:
    err = IngestionFailed("parse_error", stage=DocumentStatus.PARSING, retryable=retryable)
    with pytest.raises(ApplicationError) as caught:
        await IngestActivities(Scripted(err)).ingest_version(INP)  # type: ignore[arg-type]
    assert caught.value.non_retryable is non_retryable
    assert caught.value.message == "parse_error"


async def test_unexpected_errors_never_leak_their_message() -> None:
    secret = "CONFIDENTIAL: merger terms in page 3"  # noqa: S105 - test data, not a credential
    with pytest.raises(ApplicationError) as caught:
        await IngestActivities(Scripted(RuntimeError(secret))).ingest_version(INP)  # type: ignore[arg-type]
    assert secret not in str(caught.value) and caught.value.message == "unexpected_error"
    assert caught.value.cause is None  # `from None`: the original text is not chained either


def test_workflow_id_is_one_per_tenant_document_version() -> None:
    assert workflow_id("tenant-a", "doc_1", 3) == "ingest:tenant-a:doc_1:v3"
    assert workflow_id("tenant-b", "doc_1", 3) != workflow_id("tenant-a", "doc_1", 3)


class DownClient:
    async def start_workflow(self, *args: object, **kwargs: object) -> None:
        raise ConnectionRefusedError("temporal is down")


async def test_an_unreachable_temporal_is_a_retryable_failure_for_the_consumer() -> None:
    trigger = TemporalIngestTrigger(DownClient(), task_queue="q", retry=RetryConfig())  # type: ignore[arg-type]
    with pytest.raises(ProcessingFailure) as caught:
        await trigger.start("tenant-a", "doc_1", 3)
    assert caught.value.reason == "workflow_unavailable" and caught.value.retryable
