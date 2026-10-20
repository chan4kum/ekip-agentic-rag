"""Audit logger implementations. The event type is sanitized (see ekip_contracts.audit), so a
sink only ever receives short safe tokens."""

from ekip_contracts.audit import AuditEvent
from ekip_observability.logging import get_logger

_log = get_logger("ekip.audit")


class StructlogAuditLogger:
    """One structured JSON line per event on the `ekip.audit` logger."""

    def record(self, event: AuditEvent) -> None:
        _log.info(
            "audit",
            audit_action=event.action,
            outcome=event.outcome,
            tenant_id=event.tenant_id,
            subject=event.subject,
            resource=event.resource,
            audit_request_id=event.request_id,
            codes=list(event.codes),
            policy_version=event.policy_version,
        )


class InMemoryAuditLogger:
    """Collects events; for tests."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class NullAuditLogger:
    def record(self, event: AuditEvent) -> None:
        return None
