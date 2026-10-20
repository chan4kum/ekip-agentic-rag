import pytest
from ekip_tools import ApprovalError, ApprovalService, InMemoryReplayGuard, TokenSigner
from tools_testkit import ALICE, BOB_ADMIN, KEY, OTHER_TENANT, FakeClock, write_spec

SPEC = write_spec()
ARGS = {"text": "open a ticket"}


def service(clock: FakeClock | None = None) -> tuple[ApprovalService, FakeClock]:
    clock = clock or FakeClock()
    return ApprovalService(TokenSigner(KEY, clock=clock), InMemoryReplayGuard()), clock


async def test_approved_call_is_consumed_once() -> None:
    svc, _ = service()
    req = svc.request(ALICE, SPEC, ARGS)
    grant = svc.issue(req, BOB_ADMIN)
    rid, approver = await svc.consume(grant, ALICE, SPEC, ARGS)
    assert (rid, approver) == (req.request_id, "u-bob")
    with pytest.raises(ApprovalError, match="already_used"):
        await svc.consume(grant, ALICE, SPEC, ARGS)


async def test_changing_arguments_after_approval_invalidates_the_grant() -> None:
    svc, _ = service()
    grant = svc.issue(svc.request(ALICE, SPEC, ARGS), BOB_ADMIN)
    with pytest.raises(ApprovalError, match="arguments_changed"):
        await svc.consume(grant, ALICE, SPEC, {"text": "open a DIFFERENT ticket"})


async def test_a_grant_belongs_to_its_requester_tool_and_tenant() -> None:
    svc, _ = service()
    grant = svc.issue(svc.request(ALICE, SPEC, ARGS), BOB_ADMIN)
    with pytest.raises(ApprovalError, match="grant_mismatch"):
        await svc.consume(grant, BOB_ADMIN, SPEC, ARGS)
    with pytest.raises(ApprovalError, match="grant_mismatch"):
        await svc.consume(grant, ALICE, write_spec("other_write"), ARGS)


def test_only_a_different_approver_in_the_same_tenant_can_approve() -> None:
    svc, _ = service()
    req = svc.request(BOB_ADMIN, SPEC, ARGS)
    with pytest.raises(ApprovalError, match="self_approval"):
        svc.issue(req, BOB_ADMIN)
    req = svc.request(ALICE, SPEC, ARGS)
    with pytest.raises(ApprovalError, match="not_an_approver"):
        svc.issue(req, ALICE)
    with pytest.raises(ApprovalError, match="cross_tenant"):
        svc.issue(req, OTHER_TENANT)


async def test_grants_and_requests_expire() -> None:
    svc, clock = service()
    req = svc.request(ALICE, SPEC, ARGS)
    grant = svc.issue(req, BOB_ADMIN)
    clock.advance(10_000)
    with pytest.raises(ApprovalError, match="expired"):
        await svc.consume(grant, ALICE, SPEC, ARGS)
    with pytest.raises(ApprovalError, match="request_expired"):
        svc.issue(req, BOB_ADMIN)


async def test_garbage_grants_are_rejected() -> None:
    svc, _ = service()
    for junk in ("", "x.y", "not a token"):
        with pytest.raises(ApprovalError):
            await svc.consume(junk, ALICE, SPEC, ARGS)
