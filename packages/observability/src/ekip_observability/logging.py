"""One logging pipeline: structlog is the front door, stdlib logging is bridged into it.

Everything (our code, uvicorn, httpx, asyncpg...) ends up as the same JSON line, so redaction
and the request ID apply to library logs too.
"""

import logging
import os
import sys
from collections.abc import MutableMapping
from typing import Any, TextIO

import structlog
from opentelemetry import trace
from structlog.typing import Processor

_SENSITIVE = ("authorization", "api_key", "apikey", "password", "secret", "token", "cookie")
_OWNED = "_ekip_handler"  # marks the handler we installed, so we never touch anyone else's
_QUIET_LIBRARIES = ("httpx", "httpcore", "asyncio")  # INFO lines here carry URLs / are chatty


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "***" if any(s in str(k).lower() for s in _SENSITIVE) else _redact(v)
            for k, v in value.items()
        }
    return value


def redact_sensitive(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Mask values whose key looks sensitive, including inside nested dicts."""
    redacted: MutableMapping[str, Any] = _redact(dict(event_dict))
    return redacted


def _shared_processors(service: str, environment: str) -> list[Processor]:
    def static_fields(
        _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        event_dict.setdefault("service", service)
        event_dict.setdefault("environment", environment)
        return event_dict

    def drop_color_message(
        _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        event_dict.pop("color_message", None)  # uvicorn's ANSI-coloured duplicate of the message
        return event_dict

    def add_trace_context(
        _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:  # only inside a traced request
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
        return event_dict

    return [
        drop_color_message,
        structlog.contextvars.merge_contextvars,  # request_id etc.
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        static_fields,
        add_trace_context,
        redact_sensitive,  # BEFORE rendering
        structlog.processors.format_exc_info,
    ]


def _formatter(json_logs: bool, shared: list[Processor]) -> structlog.stdlib.ProcessorFormatter:
    """Turns ANY stdlib record into our format (this is the bridge)."""
    renderer: Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[structlog.stdlib.ExtraAdder(), *shared],  # extra= fields, then redact
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = True,
    stream: TextIO | None = None,
    service: str = "ekip",
    environment: str = "local",
) -> None:
    shared = _shared_processors(service, environment)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_formatter(json_logs, shared))
    setattr(handler, _OWNED, True)

    root = logging.getLogger()
    # Replace only our own previous handler: calling this twice never duplicates lines.
    root.handlers[:] = [h for h in root.handlers if not getattr(h, _OWNED, False)] + [handler]
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers.clear()  # drop uvicorn's own plain-text handler
        uv.propagate = True  # ...and let the root (our) handler format it
    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    # uvicorn's access log prints the full URL including ?query=..., i.e. user questions.
    # Our http_request line (path only) replaces it, so silence it.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def remove_owned_handlers() -> None:
    """Detach the handler installed by configure_logging (used by tests and shutdown)."""
    root = logging.getLogger()
    root.handlers[:] = [h for h in root.handlers if not getattr(h, _OWNED, False)]


def build_formatter(*, service: str, environment: str | None = None) -> logging.Formatter:
    """Factory for uvicorn's --log-config: JSON from the very first startup line."""
    env = environment or os.environ.get("APP_ENV", "unknown")
    return _formatter(True, _shared_processors(service, env))


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
