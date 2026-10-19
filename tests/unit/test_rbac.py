from collections.abc import Iterator
from typing import Any

import pytest
from ekip_authorization.rbac import DEFAULT_POLICY, RbacAuthorizer
from ekip_contracts.authorization import Action, Resource
from ekip_contracts.identity import SecurityContext

AUTHZ = RbacAuthorizer()


def ctx(*roles: str, tenant: str = "tenant-dev") -> SecurityContext:
    return SecurityContext(subject="u1", tenant_id=tenant, roles=frozenset(roles))


def res(tenant: str = "tenant-dev") -> Resource:
    return Resource(kind="documents", tenant_id=tenant)


@pytest.mark.parametrize(
    ("roles", "action", "allowed"),
    [
        (("ekip-user",), Action.DOCUMENTS_SEARCH, True),
        (("ekip-user",), Action.DOCUMENTS_READ, True),
        (("ekip-user",), Action.ADMIN_MANAGE, False),
        (("ekip-admin",), Action.DOCUMENTS_SEARCH, True),
        (("ekip-admin",), Action.ADMIN_MANAGE, True),
        (("ekip-user", "ekip-admin"), Action.ADMIN_MANAGE, True),
        ((), Action.DOCUMENTS_SEARCH, False),
        (("unknown-role",), Action.DOCUMENTS_SEARCH, False),
        (("EKIP-USER",), Action.DOCUMENTS_SEARCH, False),  # role names are case-sensitive
        (("ekip-user ",), Action.DOCUMENTS_SEARCH, False),  # no trimming or fuzzy matching
    ],
)
async def test_role_action_matrix(roles: tuple[str, ...], action: Action, allowed: bool) -> None:
    decision = await AUTHZ.authorize(ctx(*roles), action, res())
    assert decision.allowed is allowed
    assert decision.reason == ("allowed" if allowed else "no_role_grants_action")


@pytest.mark.parametrize("roles", [("ekip-user",), ("ekip-admin",)])
async def test_cross_tenant_resource_is_denied_even_for_admins(roles: tuple[str, ...]) -> None:
    decision = await AUTHZ.authorize(
        ctx(*roles, tenant="tenant-dev"), Action.DOCUMENTS_READ, res("tenant-other")
    )
    assert (decision.allowed, decision.reason) == (False, "tenant_mismatch")


async def test_unknown_action_string_is_denied() -> None:
    decision = await AUTHZ.authorize(ctx("ekip-admin"), "documents:delete-everything", res())  # type: ignore[arg-type]
    assert (decision.allowed, decision.reason) == (False, "unknown_action")


async def test_policy_error_fails_closed() -> None:
    class Exploding(dict[str, frozenset[Action]]):
        def get(self, key: Any, default: Any = None) -> Any:
            raise RuntimeError("policy backend exploded")

    broken = RbacAuthorizer()
    broken._policy = Exploding({"ekip-admin": frozenset(Action)})  # simulate a fault at runtime
    decision = await broken.authorize(ctx("ekip-admin"), Action.DOCUMENTS_SEARCH, res())
    assert (decision.allowed, decision.reason) == (False, "policy_error")


async def test_policy_is_copied_so_later_mutation_cannot_grant_access() -> None:
    policy: dict[str, frozenset[Action]] = {"ekip-user": frozenset({Action.DOCUMENTS_SEARCH})}
    authz = RbacAuthorizer(policy)
    policy["ekip-user"] = frozenset(Action)
    decision = await authz.authorize(ctx("ekip-user"), Action.ADMIN_MANAGE, res())
    assert decision.allowed is False


def test_default_policy_grants_only_known_actions_to_known_roles() -> None:
    assert set(DEFAULT_POLICY) == {"ekip-user", "ekip-admin"}
    assert Action.ADMIN_MANAGE not in DEFAULT_POLICY["ekip-user"]


def test_every_action_is_covered_by_at_least_one_role() -> None:
    granted: Iterator[Action] = (a for actions in DEFAULT_POLICY.values() for a in actions)
    assert set(granted) == set(Action)


def test_security_context_is_immutable_and_built_from_claims() -> None:
    from ekip_contracts.identity import AccessTokenClaims
    from pydantic import ValidationError

    claims = AccessTokenClaims(
        subject="s", issuer="i", audience=("a",), issued_at=1, expires_at=2,
        tenant_id="t", roles=("r1", "r2"), authorized_party="web",
    )  # fmt: skip
    c = SecurityContext.from_claims(claims)
    assert c.roles == frozenset({"r1", "r2"}) and c.has_role("r1") and not c.has_role("r3")
    with pytest.raises(ValidationError):
        c.tenant_id = "other"  # type: ignore[misc]
