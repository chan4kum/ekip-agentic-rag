import uuid

import structlog


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind_request_context(request_id: str, **extra: str) -> None:
    """Attach values to EVERY log line in the current request/task."""
    structlog.contextvars.bind_contextvars(request_id=request_id, **extra)


def bind_principal(*, tenant_id: str, subject: str) -> None:
    """Add the authenticated caller to every later log line of this request."""
    structlog.contextvars.bind_contextvars(tenant_id=tenant_id, subject=subject)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
