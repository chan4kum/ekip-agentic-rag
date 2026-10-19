import uuid

import structlog


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind_request_context(request_id: str, **extra: str) -> None:
    """Attach values to EVERY log line in the current request/task."""
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
