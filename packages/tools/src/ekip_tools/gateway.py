"""ToolGateway: the single door between agents and tools.

Every call passes the same pipeline, in this order (cheap and least-informative first):

  registry lookup -> authorization -> argument validation -> rate limit -> approval (writes)
  -> circuit breaker -> bulkhead -> backend call (timeout, bounded retry) -> output cap/wrap
  -> audit + trace

Properties worth knowing:
- Fail closed: any doubt (authorizer error, limiter down, unknown backend) is a refusal.
- Authorization runs BEFORE validation, so a caller without access learns nothing about schemas.
- A write approval is consumed only after every other check passed, so a refused call does not
  burn it; once consumed it is single-use.
- The model supplies only the tool name and arguments. Tenant and identity come from the
  `SecurityContext`, and the backend gets a delegated credential, never the user's token.
- Backend errors are reduced to a short safe message; exception text never reaches the model.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from ekip_contracts.identity import SecurityContext
from ekip_contracts.ratelimit import RateLimiter, RateLimitUnavailableError
from ekip_contracts.tools import (
    ApprovalRequest,
    AuditRecord,
    AuditSink,
    ToolAuthorizer,
    ToolBackend,
    ToolResult,
    ToolRisk,
    ToolSpec,
    ToolStatus,
)
from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

from ekip_tools.approval import ApprovalError, ApprovalService, arguments_digest
from ekip_tools.registry import ToolRegistry
from ekip_tools.resilience import (
    Bulkhead,
    CircuitBreaker,
    ToolBackendError,
    retry_idempotent,
)
from ekip_tools.schema import validate_arguments
from ekip_tools.wrapping import wrap_untrusted

_log = logging.getLogger(__name__)
Clock = Callable[[], float]
MAX_PROBLEMS_SHOWN = 5


class GatewayConfigError(Exception):
    """Startup misconfiguration (e.g. a registry tool with no backend)."""


class _ToolRuntime:
    """Per-tool failure containment state."""

    def __init__(self, spec: ToolSpec, clock: Clock) -> None:
        self.breaker = CircuitBreaker(clock=clock)
        self.bulkhead = Bulkhead(spec.max_concurrency)


class ToolGateway:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        backends: Mapping[str, ToolBackend],
        authorizer: ToolAuthorizer,
        user_limiter: RateLimiter,
        tenant_limiter: RateLimiter,
        audit: AuditSink,
        approvals: ApprovalService | None = None,
        tracer: Tracer | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        missing = registry.backends_in_use() - set(backends)
        if missing:
            raise GatewayConfigError(f"no backend registered for: {sorted(missing)}")
        self._registry = registry
        self._backends = dict(backends)
        self._authorizer = authorizer
        self._user_limiter = user_limiter
        self._tenant_limiter = tenant_limiter
        self._audit = audit
        self._approvals = approvals
        self._tracer = tracer or trace.NoOpTracer()
        self._runtime = {s.name: _ToolRuntime(s, clock) for s in registry.enabled()}

    async def call(
        self,
        context: SecurityContext,
        name: str,
        arguments: Mapping[str, Any],
        *,
        approval: str | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        spec = self._registry.get(name)
        digest = arguments_digest(arguments)
        with self._tracer.start_as_current_span("ekip.tool.call") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", name[:64])
            result, approval_id = await self._run(context, spec, name, arguments, approval, span)
            result = result.model_copy(
                update={"duration_ms": (time.perf_counter() - started) * 1000}
            )
            span.set_attribute("ekip.tool.status", result.status.value)
            if spec is not None:
                span.set_attribute("ekip.tool.risk", spec.risk.value)
        await self._safe_audit(context, spec, result, digest, approval_id)
        return result

    # -- pipeline ---------------------------------------------------------------------------

    async def _run(
        self,
        context: SecurityContext,
        spec: ToolSpec | None,
        name: str,
        arguments: Mapping[str, Any],
        approval: str | None,
        span: Span,
    ) -> tuple[ToolResult, str | None]:
        if spec is None:
            return _refuse(name, ToolStatus.UNKNOWN_TOOL, "no such tool"), None
        if not await self._authorized(context, spec, arguments):
            return _refuse(name, ToolStatus.DENIED, "not permitted"), None
        problems = validate_arguments(spec.input_schema, arguments)
        if problems:
            shown = "; ".join(problems[:MAX_PROBLEMS_SHOWN])
            return _refuse(name, ToolStatus.INVALID_ARGUMENTS, shown), None
        limited = await self._rate_limited(context, spec)
        if limited is not None:
            return limited, None
        approval_id: str | None = None
        if spec.risk is ToolRisk.WRITE:
            gate = await self._check_approval(context, spec, arguments, approval)
            if isinstance(gate, ToolResult):
                return gate, None
            approval_id = gate
        runtime = self._runtime[spec.name]
        if not runtime.breaker.allow():
            return _refuse(name, ToolStatus.UNAVAILABLE, "temporarily unavailable"), approval_id
        if not runtime.bulkhead.try_acquire():
            return _refuse(name, ToolStatus.UNAVAILABLE, "too many concurrent calls"), approval_id
        try:
            return await self._invoke(context, spec, arguments, runtime, span), approval_id
        finally:
            runtime.bulkhead.release()

    async def _authorized(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> bool:
        try:
            decision = await self._authorizer.authorize(context, spec, arguments)
        except Exception:  # noqa: BLE001 - any authorizer failure means deny (fail closed)
            return False
        return decision.allowed

    async def _rate_limited(self, context: SecurityContext, spec: ToolSpec) -> ToolResult | None:
        keys = (
            (self._user_limiter, f"tool:u:{context.tenant_id}:{context.subject}"),
            (self._tenant_limiter, f"tool:t:{context.tenant_id}"),
        )
        try:
            for limiter, key in keys:
                decision = await limiter.check(key, cost=spec.rate_limit_cost)
                if not decision.allowed:
                    return _refuse(spec.name, ToolStatus.RATE_LIMITED, "rate limit exceeded")
        except RateLimitUnavailableError:
            return _refuse(spec.name, ToolStatus.UNAVAILABLE, "rate limiter unavailable")
        return None

    async def _check_approval(
        self,
        context: SecurityContext,
        spec: ToolSpec,
        arguments: Mapping[str, Any],
        approval: str | None,
    ) -> str | ToolResult:
        """Returns the approval request id when the call may run, else the refusal."""
        if self._approvals is None:
            return _refuse(spec.name, ToolStatus.APPROVAL_REQUIRED, "approvals not configured")
        if approval is None:
            request: ApprovalRequest = self._approvals.request(context, spec, arguments)
            return ToolResult(
                tool=spec.name,
                status=ToolStatus.APPROVAL_REQUIRED,
                detail="a human approval is required",
                approval_request=request,
            )
        try:
            request_id, _approver = await self._approvals.consume(
                approval, context, spec, arguments
            )
        except ApprovalError as exc:
            return _refuse(
                spec.name, ToolStatus.APPROVAL_REQUIRED, f"approval rejected: {exc.reason}"
            )
        return request_id

    async def _invoke(
        self,
        context: SecurityContext,
        spec: ToolSpec,
        arguments: Mapping[str, Any],
        runtime: _ToolRuntime,
        span: Span,
    ) -> ToolResult:
        backend = self._backends[spec.backend]

        async def once() -> str:
            async with asyncio.timeout(spec.timeout_s):
                return await backend.call(context, spec, arguments)

        try:
            raw = await retry_idempotent(once, retries=spec.max_retries)
        except TimeoutError:
            runtime.breaker.record_failure()
            return _refuse(spec.name, ToolStatus.TIMEOUT, "tool timed out")
        except ToolBackendError as exc:
            runtime.breaker.record_failure()
            return _refuse(spec.name, ToolStatus.ERROR, exc.safe_message[:200])
        except Exception:  # noqa: BLE001 - unknown backend bug: never leak its message
            runtime.breaker.record_failure()
            span.set_attribute("ekip.tool.unexpected_error", True)
            return _refuse(spec.name, ToolStatus.ERROR, "tool failed")
        runtime.breaker.record_success()
        content, truncated = wrap_untrusted(spec.name, raw, max_chars=spec.max_output_chars)
        return ToolResult(
            tool=spec.name, status=ToolStatus.OK, content=content, truncated=truncated
        )

    async def _safe_audit(
        self,
        context: SecurityContext,
        spec: ToolSpec | None,
        result: ToolResult,
        digest: str,
        approval_id: str | None,
    ) -> None:
        request_id = approval_id or (
            result.approval_request.request_id if result.approval_request else None
        )
        try:
            await self._audit.record(
                AuditRecord(
                    tool=result.tool[:64],
                    status=result.status,
                    subject=context.subject,
                    tenant_id=context.tenant_id,
                    arguments_digest=digest,
                    duration_ms=result.duration_ms,
                    risk=spec.risk if spec else None,
                    approval_request_id=request_id,
                    detail=result.detail[:200],
                )
            )
        except Exception:  # noqa: BLE001 - an audit outage must not mask the tool result
            _log.error("tool audit sink failed", extra={"tool": result.tool[:64]})


def _refuse(tool: str, status: ToolStatus, detail: str) -> ToolResult:
    return ToolResult(tool=tool[:64], status=status, detail=detail)
