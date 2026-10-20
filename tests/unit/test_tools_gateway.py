import asyncio
from typing import Any

import pytest
from ekip_contracts.ratelimit import RateLimitDecision, RateLimitUnavailableError
from ekip_contracts.tools import ToolDecision, ToolStatus
from ekip_tools import (
    GatewayConfigError,
    RoleToolAuthorizer,
    ToolBackendError,
    ToolGateway,
    ToolRegistry,
    arguments_digest,
)
from tools_testkit import (
    ALICE,
    BOB_ADMIN,
    NOBODY,
    OTHER_TENANT,
    FakeBackend,
    Harness,
    make_spec,
    write_spec,
)

OK_ARGS = {"text": "hello"}


async def test_happy_path_wraps_output_and_audits() -> None:
    h = Harness([make_spec()])
    result = await h.gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert result.status is ToolStatus.OK
    assert result.content.startswith('<tool_output tool="echo_tool">')
    assert "ok:hello" in result.content
    (record,) = h.audit.records
    assert (record.tool, record.status, record.subject) == ("echo_tool", ToolStatus.OK, "u-alice")
    assert record.arguments_digest == arguments_digest(OK_ARGS)
    assert "hello" not in record.model_dump_json()  # arguments are digested, never logged


async def test_unknown_and_disabled_tools() -> None:
    h = Harness([make_spec(), make_spec("off_tool", enabled=False)])
    for name in ("nope_tool", "off_tool", "../etc/passwd"):
        result = await h.gateway.call(ALICE, name, OK_ARGS)
        assert result.status is ToolStatus.UNKNOWN_TOOL
    assert h.backend.calls == []


async def test_unauthorized_callers_are_denied_before_validation() -> None:
    h = Harness([make_spec()])
    result = await h.gateway.call(NOBODY, "echo_tool", {"bogus": 1})
    assert result.status is ToolStatus.DENIED  # not INVALID_ARGUMENTS: no schema oracle
    assert h.backend.calls == []


async def test_authorizer_errors_fail_closed() -> None:
    class Broken:
        async def authorize(self, *a: Any, **k: Any) -> ToolDecision:
            raise RuntimeError("policy down")

    h = Harness([make_spec()])
    gateway = ToolGateway(
        registry=ToolRegistry([make_spec()]),
        backends={"fake": h.backend},
        authorizer=Broken(),
        user_limiter=h.user_limiter,
        tenant_limiter=h.tenant_limiter,
        audit=h.audit,
    )
    assert (await gateway.call(ALICE, "echo_tool", OK_ARGS)).status is ToolStatus.DENIED


@pytest.mark.parametrize(
    "arguments",
    [{}, {"text": 5}, {"text": "x" * 51}, {"text": "ok", "tenant_id": "tenant-b"}],
)
async def test_invalid_arguments_never_reach_the_backend(arguments: dict[str, Any]) -> None:
    h = Harness([make_spec()])
    result = await h.gateway.call(ALICE, "echo_tool", arguments)
    assert result.status is ToolStatus.INVALID_ARGUMENTS
    assert h.backend.calls == []


async def test_rate_limit_per_user_and_cost() -> None:
    h = Harness([make_spec(rate_limit_cost=2)], capacity=4)
    statuses = [(await h.gateway.call(ALICE, "echo_tool", OK_ARGS)).status for _ in range(3)]
    assert statuses == [ToolStatus.OK, ToolStatus.OK, ToolStatus.RATE_LIMITED]
    h.clock.advance(60)
    assert (await h.gateway.call(ALICE, "echo_tool", OK_ARGS)).status is ToolStatus.OK


async def test_limiter_outage_fails_closed() -> None:
    class Down:
        async def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
            raise RateLimitUnavailableError

    h = Harness([make_spec()])
    gateway = ToolGateway(
        registry=ToolRegistry([make_spec()]),
        backends={"fake": h.backend},
        authorizer=RoleToolAuthorizer(),
        user_limiter=Down(),
        tenant_limiter=h.tenant_limiter,
        audit=h.audit,
    )
    result = await gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert result.status is ToolStatus.UNAVAILABLE
    assert h.backend.calls == []


async def test_timeout_and_backend_errors_do_not_leak_internals() -> None:
    async def slow(_: Any) -> str:
        await asyncio.sleep(5)
        return "late"

    def boom(_: Any) -> str:
        raise RuntimeError("password=hunter2 at /srv/internal")

    def known(_: Any) -> str:
        raise ToolBackendError("upstream error")

    h = Harness([make_spec("slow_tool", timeout_s=0.05)], FakeBackend(slow))
    assert (await h.gateway.call(ALICE, "slow_tool", OK_ARGS)).status is ToolStatus.TIMEOUT

    h = Harness([make_spec()], FakeBackend(boom))
    result = await h.gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert result.status is ToolStatus.ERROR
    assert "hunter2" not in result.model_dump_json()

    h = Harness([make_spec()], FakeBackend(known))
    result = await h.gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert (result.status, result.detail) == (ToolStatus.ERROR, "upstream error")


async def test_output_is_truncated() -> None:
    h = Harness([make_spec(max_output_chars=100)], FakeBackend(lambda _: "y" * 5000))
    result = await h.gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert result.truncated
    assert result.content.count("y") == 100


async def test_circuit_opens_after_repeated_failures_and_recovers() -> None:
    state = {"fail": True}

    def flaky(_: Any) -> str:
        if state["fail"]:
            raise ToolBackendError("down")
        return "fine"

    h = Harness([make_spec()], FakeBackend(flaky))
    for _ in range(5):
        assert (await h.gateway.call(ALICE, "echo_tool", OK_ARGS)).status is ToolStatus.ERROR
    blocked = await h.gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert blocked.status is ToolStatus.UNAVAILABLE
    assert len(h.backend.calls) == 5  # the sixth call never reached the backend
    state["fail"] = False
    h.clock.advance(31)
    assert (await h.gateway.call(ALICE, "echo_tool", OK_ARGS)).status is ToolStatus.OK


async def test_bulkhead_rejects_excess_concurrency() -> None:
    gate = asyncio.Event()

    async def hold(_: Any) -> str:
        await gate.wait()
        return "released"

    h = Harness([make_spec(max_concurrency=2, timeout_s=5)], FakeBackend(hold))
    running = [asyncio.create_task(h.gateway.call(ALICE, "echo_tool", OK_ARGS)) for _ in range(2)]
    await asyncio.sleep(0.01)
    rejected = await h.gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert rejected.status is ToolStatus.UNAVAILABLE
    gate.set()
    assert all(r.status is ToolStatus.OK for r in await asyncio.gather(*running))


async def test_idempotent_tools_retry_transient_failures() -> None:
    attempts = {"n": 0}

    def flaky(_: Any) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ToolBackendError("blip", retryable=True)
        return "second try"

    h = Harness([make_spec(idempotent=True, max_retries=1)], FakeBackend(flaky))
    result = await h.gateway.call(ALICE, "echo_tool", OK_ARGS)
    assert result.status is ToolStatus.OK
    assert attempts["n"] == 2


async def test_write_tools_need_a_matching_single_use_approval() -> None:
    h = Harness([write_spec()])
    first = await h.gateway.call(ALICE, "write_tool", OK_ARGS)
    assert first.status is ToolStatus.APPROVAL_REQUIRED
    assert first.approval_request is not None
    assert first.approval_request.arguments_digest == arguments_digest(OK_ARGS)
    assert h.backend.calls == []

    grant = h.approvals.issue(first.approval_request, BOB_ADMIN)
    tampered = await h.gateway.call(ALICE, "write_tool", {"text": "different"}, approval=grant)
    assert tampered.status is ToolStatus.APPROVAL_REQUIRED
    assert h.backend.calls == []

    done = await h.gateway.call(ALICE, "write_tool", OK_ARGS, approval=grant)
    assert done.status is ToolStatus.OK
    assert len(h.backend.calls) == 1
    replay = await h.gateway.call(ALICE, "write_tool", OK_ARGS, approval=grant)
    assert replay.status is ToolStatus.APPROVAL_REQUIRED
    assert len(h.backend.calls) == 1
    assert any(r.approval_request_id == first.approval_request.request_id for r in h.audit.records)


async def test_refused_calls_do_not_burn_an_approval() -> None:
    h = Harness([write_spec()], capacity=1)
    request = (await h.gateway.call(ALICE, "write_tool", OK_ARGS)).approval_request
    assert request is not None
    grant = h.approvals.issue(request, BOB_ADMIN)
    h.clock.advance(0)  # bucket empty after the first call (cost 1, capacity 1)
    limited = await h.gateway.call(ALICE, "write_tool", OK_ARGS, approval=grant)
    assert limited.status is ToolStatus.RATE_LIMITED
    h.clock.advance(5)
    assert (
        await h.gateway.call(ALICE, "write_tool", OK_ARGS, approval=grant)
    ).status is ToolStatus.OK


async def test_write_tools_without_an_approval_service_are_refused() -> None:
    h = Harness([write_spec()])
    gateway = ToolGateway(
        registry=ToolRegistry([write_spec()]),
        backends={"fake": h.backend},
        authorizer=RoleToolAuthorizer(),
        user_limiter=h.user_limiter,
        tenant_limiter=h.tenant_limiter,
        audit=h.audit,
    )
    result = await gateway.call(ALICE, "write_tool", OK_ARGS)
    assert result.status is ToolStatus.APPROVAL_REQUIRED
    assert result.approval_request is None
    assert h.backend.calls == []


async def test_a_broken_audit_sink_does_not_mask_the_result() -> None:
    class BrokenAudit:
        async def record(self, entry: Any) -> None:
            raise RuntimeError("audit down")

    h = Harness([make_spec()])
    gateway = ToolGateway(
        registry=ToolRegistry([make_spec()]),
        backends={"fake": h.backend},
        authorizer=RoleToolAuthorizer(),
        user_limiter=h.user_limiter,
        tenant_limiter=h.tenant_limiter,
        audit=BrokenAudit(),
    )
    assert (await gateway.call(ALICE, "echo_tool", OK_ARGS)).status is ToolStatus.OK


def test_every_enabled_tool_needs_a_backend_at_startup() -> None:
    h = Harness([make_spec()])
    with pytest.raises(GatewayConfigError):
        ToolGateway(
            registry=ToolRegistry([make_spec(backend="missing")]),
            backends={"fake": h.backend},
            authorizer=RoleToolAuthorizer(),
            user_limiter=h.user_limiter,
            tenant_limiter=h.tenant_limiter,
            audit=h.audit,
        )


async def test_tenant_rate_limit_is_shared_across_users() -> None:
    h = Harness([make_spec()], capacity=2)
    other = OTHER_TENANT.model_copy(update={"tenant_id": "tenant-a", "subject": "u-carol"})
    assert (await h.gateway.call(ALICE, "echo_tool", OK_ARGS)).ok
    assert (await h.gateway.call(other, "echo_tool", OK_ARGS)).ok
    assert (await h.gateway.call(BOB_ADMIN, "echo_tool", OK_ARGS)).status is ToolStatus.RATE_LIMITED
