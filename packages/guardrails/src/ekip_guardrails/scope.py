"""Per-request guard state, shared between the guarded retriever and the guarded model.

The retriever learns which source URLs are legitimate (the citation set); the model wrapper needs
that set to tell a real citation link from an exfiltration URL. A ContextVar holding ONE mutable
object works across LangGraph's tasks because every task sees the same object.

Set it once per request, from the verified SecurityContext - never from request data.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class GuardScope:
    tenant_id: str
    subject: str = ""
    request_id: str = ""
    allowed_urls: set[str] = field(default_factory=set)


_current: ContextVar[GuardScope | None] = ContextVar("ekip_guard_scope", default=None)


def current_scope() -> GuardScope | None:
    return _current.get()


@contextmanager
def guard_scope(tenant_id: str, *, subject: str = "", request_id: str = "") -> Iterator[GuardScope]:
    scope = GuardScope(tenant_id=tenant_id, subject=subject, request_id=request_id)
    token = _current.set(scope)
    try:
        yield scope
    finally:
        _current.reset(token)
