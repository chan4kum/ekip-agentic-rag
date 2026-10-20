"""Shared fakes for the tool-gateway tests (no network, no sleeping)."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ToolRisk, ToolSpec
from ekip_ratelimit.memory import InMemoryRateLimiter
from ekip_tools import (
    ApprovalService,
    InMemoryAuditSink,
    InMemoryReplayGuard,
    RoleToolAuthorizer,
    TokenSigner,
    ToolGateway,
    ToolRegistry,
)

KEY = b"k" * 32
ALICE = SecurityContext(
    subject="u-alice", tenant_id="tenant-a", roles=frozenset({"ekip-user"}), principal="alice"
)
BOB_ADMIN = SecurityContext(
    subject="u-bob", tenant_id="tenant-a", roles=frozenset({"ekip-admin", "ekip-user"})
)
OTHER_TENANT = SecurityContext(
    subject="u-eve", tenant_id="tenant-b", roles=frozenset({"ekip-user", "ekip-admin"})
)
NOBODY = SecurityContext(subject="u-none", tenant_id="tenant-a", roles=frozenset())


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_spec(name: str = "echo_tool", **overrides: Any) -> ToolSpec:
    fields: dict[str, Any] = {
        "name": name,
        "description": "test tool",
        "backend": "fake",
        "required_roles": ("ekip-user",),
        "input_schema": {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string", "maxLength": 50}},
        },
        "timeout_s": 1.0,
    }
    fields.update(overrides)
    return ToolSpec(**fields)


def write_spec(name: str = "write_tool", **overrides: Any) -> ToolSpec:
    return make_spec(name, risk=ToolRisk.WRITE, **overrides)


class FakeBackend:
    """Records calls; `behavior` may return text or raise."""

    def __init__(
        self, behavior: Callable[[Mapping[str, Any]], Awaitable[str] | str] | None = None
    ) -> None:
        self.calls: list[tuple[SecurityContext, Mapping[str, Any]]] = []
        self._behavior = behavior
        self.in_flight = 0

    async def call(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> str:
        self.calls.append((context, arguments))
        self.in_flight += 1
        try:
            if self._behavior is None:
                return f"ok:{arguments.get('text', '')}"
            result = self._behavior(arguments)
            if asyncio.iscoroutine(result):
                return str(await result)
            return str(result)
        finally:
            self.in_flight -= 1


class Harness:
    def __init__(
        self,
        specs: list[ToolSpec],
        backend: FakeBackend | None = None,
        *,
        capacity: int = 100,
        clock: FakeClock | None = None,
    ) -> None:
        self.clock = clock or FakeClock()
        self.backend = backend or FakeBackend()
        self.audit = InMemoryAuditSink()
        self.signer = TokenSigner(KEY, clock=self.clock)
        self.approvals = ApprovalService(self.signer, InMemoryReplayGuard())
        self.user_limiter = InMemoryRateLimiter(
            capacity=capacity, refill_per_second=1.0, clock=self.clock
        )
        self.tenant_limiter = InMemoryRateLimiter(
            capacity=capacity, refill_per_second=1.0, clock=self.clock
        )
        self.gateway = ToolGateway(
            registry=ToolRegistry(specs),
            backends={"fake": self.backend},
            authorizer=RoleToolAuthorizer(),
            user_limiter=self.user_limiter,
            tenant_limiter=self.tenant_limiter,
            audit=self.audit,
            approvals=self.approvals,
            clock=self.clock,
        )
