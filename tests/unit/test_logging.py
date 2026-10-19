import io
import json
from collections.abc import Iterator
from typing import Any

import pytest
import structlog
from ekip_observability.context import (
    bind_request_context,
    clear_request_context,
    new_request_id,
)
from ekip_observability.logging import configure_logging, get_logger


@pytest.fixture
def buf() -> Iterator[io.StringIO]:
    structlog.reset_defaults()
    clear_request_context()
    stream = io.StringIO()
    configure_logging("INFO", json_logs=True, stream=stream)
    yield stream
    clear_request_context()
    structlog.reset_defaults()


def lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_json_line_has_event_level_and_iso_timestamp(buf: io.StringIO) -> None:
    get_logger().info("hello", answer=42)
    (rec,) = lines(buf)
    assert rec["event"] == "hello"
    assert rec["level"] == "info"
    assert rec["answer"] == 42
    assert "T" in rec["timestamp"]


def test_request_id_is_attached_then_cleared(buf: io.StringIO) -> None:
    log = get_logger()
    bind_request_context("abc")
    log.info("inside")
    clear_request_context()
    log.info("outside")
    inside, outside = lines(buf)
    assert inside["request_id"] == "abc"
    assert "request_id" not in outside


def test_sensitive_values_are_redacted_including_nested(buf: io.StringIO) -> None:
    get_logger().info(
        "call", authorization="Bearer SECRET", payload={"nested": {"api_key": "SECRET"}}
    )
    (rec,) = lines(buf)
    assert rec["authorization"] == "***"
    assert rec["payload"]["nested"]["api_key"] == "***"
    assert "SECRET" not in buf.getvalue()


def test_debug_is_suppressed_at_info(buf: io.StringIO) -> None:
    get_logger().debug("noisy")
    assert lines(buf) == []


def test_exception_is_rendered(buf: io.StringIO) -> None:
    log = get_logger()
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("failed")
    (rec,) = lines(buf)
    assert "ValueError" in rec["exception"]


def test_request_ids_are_unique_hex() -> None:
    a, b = new_request_id(), new_request_id()
    assert a != b
    assert len(a) == 32
    int(a, 16)
