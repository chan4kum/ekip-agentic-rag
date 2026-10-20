"""Audit sinks. Entries carry an arguments digest, never arguments or outputs (I-8)."""

import logging

from ekip_contracts.tools import AuditRecord

_log = logging.getLogger("ekip.tools.audit")


class LoggingAuditSink:
    async def record(self, entry: AuditRecord) -> None:
        _log.info(
            "tool_call",
            extra={
                "tool": entry.tool,
                "status": entry.status.value,
                "subject": entry.subject,
                "tenant_id": entry.tenant_id,
                "arguments_digest": entry.arguments_digest,
                "duration_ms": round(entry.duration_ms, 2),
                "risk": entry.risk.value if entry.risk else None,
                "approval_request_id": entry.approval_request_id,
            },
        )


class InMemoryAuditSink:
    """For tests and local runs."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def record(self, entry: AuditRecord) -> None:
        self.records.append(entry)
