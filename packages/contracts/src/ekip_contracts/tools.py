"""Tool contracts: what a callable tool is, what a call returns, and the ports around the gateway.

The agent never calls a backend directly. It asks a `ToolInvoker` (the gateway) to run a tool by
name; the gateway authorizes, validates, limits and audits the call (invariants I-1, I-2). The
model can influence only the tool NAME and ARGUMENTS, both of which are checked against a
server-side registry. A tool's output is data, never an instruction (see `ToolResult`).
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ekip_contracts.identity import SecurityContext

TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{2,63}")
ROLE_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,64}")

# Markers around tool output so the model can tell data from instructions. The gateway escapes
# these markers inside the payload, so a tool result cannot close the wrapper early.
UNTRUSTED_OPEN = "<tool_output"
UNTRUSTED_CLOSE = "</tool_output>"


class ToolRisk(StrEnum):
    READ = "read"  # no side effects
    WRITE = "write"  # changes something: needs a human approval for every call


class ToolStatus(StrEnum):
    OK = "ok"
    UNKNOWN_TOOL = "unknown_tool"
    DENIED = "denied"  # the caller may not use this tool
    INVALID_ARGUMENTS = "invalid_arguments"
    RATE_LIMITED = "rate_limited"
    APPROVAL_REQUIRED = "approval_required"  # a write tool without a valid approval
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"  # circuit open, bulkhead full, or backend down
    ERROR = "error"  # the backend failed in a way that is not worth retrying


class ToolSpec(BaseModel):
    """Everything the gateway knows about one tool. Loaded from `tools/registry.yaml`, reviewed
    in git; the model never edits it. `input_schema` is a deliberately small JSON-schema subset
    (see ekip_tools.schema) so validation is deterministic and cheap."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = Field(min_length=1, max_length=500)
    version: str = "1"
    risk: ToolRisk = ToolRisk.READ
    backend: str = Field(min_length=1, max_length=64)  # key of the registered backend
    required_roles: tuple[str, ...] = Field(min_length=1)  # caller needs ANY of these
    input_schema: dict[str, Any]
    timeout_s: float = Field(default=10.0, gt=0, le=120)
    max_output_chars: int = Field(default=8000, ge=100, le=100_000)
    idempotent: bool = False  # only idempotent tools are retried
    max_retries: int = Field(default=0, ge=0, le=3)
    rate_limit_cost: int = Field(default=1, ge=1, le=100)
    max_concurrency: int = Field(default=8, ge=1, le=1000)
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not TOOL_NAME.fullmatch(value):
            raise ValueError("tool name must be lower_snake_case, 3-64 chars")
        return value

    @field_validator("required_roles")
    @classmethod
    def _check_roles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not all(ROLE_NAME.fullmatch(role) for role in value):
            raise ValueError("invalid role name")
        return value

    @model_validator(mode="after")
    def _writes_are_never_retried(self) -> "ToolSpec":
        if self.risk is ToolRisk.WRITE and (self.idempotent or self.max_retries):
            raise ValueError("write tools are never retried")
        if self.max_retries and not self.idempotent:
            raise ValueError("only idempotent tools may be retried")
        return self


class ToolResult(BaseModel):
    """What the agent gets back. `content` is UNTRUSTED text already wrapped and escaped by the
    gateway; treat it as evidence, never as instructions. It is empty unless `status` is OK."""

    model_config = ConfigDict(frozen=True)

    tool: str
    status: ToolStatus
    content: str = ""
    truncated: bool = False
    duration_ms: float = 0.0
    detail: str = ""  # short, safe reason for non-OK statuses (no backend internals)
    approval_request: "ApprovalRequest | None" = None

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK


class ApprovalRequest(BaseModel):
    """What a human is asked to approve: this tool, with exactly these arguments (by digest)."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    tool: str
    arguments_digest: str  # sha256 of the canonical JSON arguments
    requested_by: str  # SecurityContext.subject
    tenant_id: str
    summary: str  # human-readable one-liner built by the gateway, not by the model
    expires_at: int  # unix seconds


class ToolDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str = ""


class ToolInvoker(Protocol):
    """Port the agents use. Implemented by ekip_tools.ToolGateway."""

    async def call(
        self,
        context: SecurityContext,
        name: str,
        arguments: Mapping[str, Any],
        *,
        approval: str | None = None,
    ) -> ToolResult: ...


class ToolAuthorizer(Protocol):
    async def authorize(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> ToolDecision:
        """Anything uncertain is a deny (fail closed)."""
        ...


class ToolBackend(Protocol):
    """Runs a tool. The gateway has already authorized and validated; the backend must STILL
    enforce tenant isolation itself (defence in depth), using `context`, never the arguments."""

    async def call(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> str: ...


class AuditRecord(BaseModel):
    """One line per tool call. Arguments are recorded only as a digest, outputs not at all."""

    model_config = ConfigDict(frozen=True)

    tool: str
    status: ToolStatus
    subject: str
    tenant_id: str
    arguments_digest: str
    duration_ms: float
    risk: ToolRisk | None = None
    approval_request_id: str | None = None
    detail: str = ""


class AuditSink(Protocol):
    async def record(self, entry: AuditRecord) -> None: ...


ToolResult.model_rebuild()
