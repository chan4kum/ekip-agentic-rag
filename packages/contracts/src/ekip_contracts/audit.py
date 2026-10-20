"""Port for security audit events (invariant I-11).

An event says WHO did WHAT to WHICH resource with WHICH outcome. It never carries prompt, document
or answer text: every string field is restricted to a short safe-token alphabet, and anything
else is replaced by "invalid" so a caller bug cannot smuggle content into the audit trail.
"""

import re
from dataclasses import dataclass
from typing import Literal, Protocol

AuditOutcome = Literal["allowed", "masked", "blocked", "denied", "error"]

_SAFE = re.compile(r"[A-Za-z0-9_.:@/\-]{0,128}")


def _safe(value: str) -> str:
    return value if _SAFE.fullmatch(value) else "invalid"


@dataclass(frozen=True)
class AuditEvent:
    action: str  # fixed vocabulary, e.g. "guardrail.input", "guardrail.output", "tool.call"
    outcome: AuditOutcome
    tenant_id: str = ""
    subject: str = ""
    resource: str = ""  # an id (tool name, document id) - never content
    request_id: str = ""
    codes: tuple[str, ...] = ()  # fixed finding codes
    policy_version: str = ""

    def __post_init__(self) -> None:
        for name in ("action", "tenant_id", "subject", "resource", "request_id", "policy_version"):
            object.__setattr__(self, name, _safe(getattr(self, name)))
        object.__setattr__(self, "codes", tuple(_safe(c) for c in self.codes))


class AuditLogger(Protocol):
    def record(self, event: AuditEvent) -> None:
        """Persist the event. Must not raise for a well-formed event; a failing audit sink must
        never turn a denial into an allow."""
        ...
