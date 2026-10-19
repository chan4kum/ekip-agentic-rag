"""stdlib logging (uvicorn, httpx, asyncpg...) must flow through the same pipeline."""

import io
import json
import logging
import logging.config
from pathlib import Path
from typing import Any

import pytest
from ekip_observability.context import bind_request_context, clear_request_context
from ekip_observability.logging import build_formatter, configure_logging, get_logger

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def buf() -> io.StringIO:
    stream = io.StringIO()
    configure_logging("INFO", json_logs=True, stream=stream, service="svc", environment="test")
    return stream


def lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_stdlib_record_becomes_one_json_line_with_standard_fields(buf: io.StringIO) -> None:
    logging.getLogger("uvicorn.error").error("Application startup failed")
    (rec,) = lines(buf)
    assert rec["event"] == "Application startup failed"
    assert rec["level"] == "error"
    assert rec["logger"] == "uvicorn.error"
    assert rec["service"] == "svc"
    assert rec["environment"] == "test"
    assert "T" in rec["timestamp"]


def test_structlog_lines_carry_logger_service_and_environment(buf: io.StringIO) -> None:
    get_logger("ekip.http").info("hello")
    (rec,) = lines(buf)
    assert rec["logger"] == "ekip.http"
    assert (rec["service"], rec["environment"]) == ("svc", "test")


def test_stdlib_exception_is_rendered(buf: io.StringIO) -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("lib").exception("failed")
    (rec,) = lines(buf)
    assert "ValueError" in rec["exception"]


def test_secrets_in_stdlib_extra_fields_are_redacted(buf: io.StringIO) -> None:
    logging.getLogger("lib").info("call", extra={"api_key": "SECRET-123", "ok": 1})
    (rec,) = lines(buf)
    assert rec["api_key"] == "***"
    assert rec["ok"] == 1
    assert "SECRET-123" not in buf.getvalue()


def test_newline_in_a_message_cannot_forge_a_second_log_line(buf: io.StringIO) -> None:
    logging.getLogger("lib").info("user said hi\n" + '{"event":"forged"}')
    assert len(buf.getvalue().splitlines()) == 1
    assert len(lines(buf)) == 1


def test_request_id_reaches_stdlib_lines(buf: io.StringIO) -> None:
    bind_request_context("rid-9")
    logging.getLogger("lib").info("inside request")
    clear_request_context()
    (rec,) = lines(buf)
    assert rec["request_id"] == "rid-9"


def test_configuring_twice_does_not_duplicate_lines() -> None:
    first, second = io.StringIO(), io.StringIO()
    configure_logging(stream=first)
    configure_logging(stream=second)
    get_logger().info("once")
    logging.getLogger("lib").info("once more")
    assert first.getvalue() == ""
    assert len(second.getvalue().splitlines()) == 2


def test_foreign_handlers_are_left_alone() -> None:
    other = logging.NullHandler()
    logging.getLogger().addHandler(other)
    try:
        configure_logging(stream=io.StringIO())
        assert other in logging.getLogger().handlers
    finally:
        logging.getLogger().removeHandler(other)


def test_chatty_libraries_are_quiet_at_info_but_warnings_pass(buf: io.StringIO) -> None:
    logging.getLogger("httpx").info('HTTP Request: GET https://x/?q=private "200 OK"')
    logging.getLogger("httpx").warning("retrying")
    (rec,) = lines(buf)
    assert rec["event"] == "retrying"
    assert "private" not in buf.getvalue()


def test_uvicorn_loggers_use_the_root_handler(buf: io.StringIO) -> None:
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        assert lg.handlers == []
        assert lg.propagate is True


def test_console_mode_still_works_for_local_development() -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_logs=False, stream=stream)
    get_logger("x").info("pretty")
    assert "pretty" in stream.getvalue()
    assert not stream.getvalue().lstrip().startswith("{")


def test_uvicorn_log_config_file_produces_json(capsys: pytest.CaptureFixture[str]) -> None:
    config = json.loads((ROOT / "apps/api/uvicorn-log.json").read_text())
    logging.config.dictConfig(config)
    logging.getLogger("uvicorn.error").info("Started server process")
    rec = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rec["event"] == "Started server process"
    assert rec["service"] == "ekip-api"
    # dictConfig attached a handler to the 'uvicorn' logger only; clean up for other tests
    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn").propagate = True


def test_build_formatter_returns_a_formatter() -> None:
    assert isinstance(build_formatter(service="s", environment="e"), logging.Formatter)


def test_uvicorn_access_log_is_silenced_because_it_prints_query_strings(
    buf: io.StringIO,
) -> None:
    logging.getLogger("uvicorn.access").info(
        '1.2.3.4 - "GET /api/v1/search?q=private HTTP/1.1" 200'
    )
    assert buf.getvalue() == ""


def test_uvicorn_color_message_duplicate_is_dropped(buf: io.StringIO) -> None:
    logging.getLogger("uvicorn.error").info(
        "Started server process [%d]", 1, extra={"color_message": "\x1b[36mcoloured\x1b[0m"}
    )
    (rec,) = lines(buf)
    assert "color_message" not in rec
    assert "\x1b" not in buf.getvalue()
