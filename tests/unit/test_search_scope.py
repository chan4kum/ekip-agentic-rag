"""SearchScope and the token claims that feed it (principal, groups): bad input fails closed."""

import pytest
from ekip_contracts.documents import ContractError
from ekip_contracts.identity import InvalidTokenError, SecurityContext
from ekip_contracts.retrieval import SearchScope
from ekip_identity.validator import JwtTokenValidator
from ekip_search import tenant_and_acl_filter
from idp_testkit import TestIdp, claims


def test_scope_is_built_from_a_verified_context() -> None:
    ctx = SecurityContext(
        subject="u-1", tenant_id="tenant-a", principal="alice", groups=frozenset({"eng"})
    )
    scope = SearchScope.from_context(ctx)
    assert (scope.tenant_id, scope.user, scope.groups) == ("tenant-a", "alice", frozenset({"eng"}))
    assert scope.matches_any_acl


def test_a_context_without_principal_or_groups_matches_no_acl() -> None:
    ctx = SecurityContext(subject="u-1", tenant_id="tenant-a")
    scope = SearchScope.from_context(ctx)
    assert scope.user == "" and not scope.matches_any_acl


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tenant_id": ""},
        {"tenant_id": "bad tenant"},
        {"tenant_id": "t", "user": "has space"},
        {"tenant_id": "t", "groups": frozenset({"ok", "no good"})},
    ],
)
def test_invalid_scopes_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        SearchScope(**kwargs)  # type: ignore[arg-type]


def test_the_filter_for_a_caller_without_identity_is_match_none() -> None:
    assert {"match_none": {}} in tenant_and_acl_filter("tenant-a", "", set())


def test_the_filter_only_lists_identities_the_caller_has() -> None:
    [_, acl] = tenant_and_acl_filter("tenant-a", "alice", set())
    assert acl["bool"]["should"] == [{"term": {"allowed_users": "alice"}}]
    [_, acl] = tenant_and_acl_filter("tenant-a", "", {"eng", "ops"})
    assert acl["bool"]["should"] == [{"terms": {"allowed_groups": ["eng", "ops"]}}]


async def test_claims_carry_principal_and_groups() -> None:
    idp = TestIdp()
    c = await idp.validator().validate(idp.token(principal="alice", groups=["ops", "eng"]))
    assert c.principal == "alice" and c.groups == ("eng", "ops")


async def test_missing_claims_default_to_nothing() -> None:
    idp = TestIdp()
    c = await idp.validator().validate(idp.token(principal=None, groups=None))
    assert c.principal is None and c.groups == ()


async def test_malformed_principal_and_groups_are_dropped_not_guessed() -> None:
    idp = TestIdp()
    token = idp.token(principal="has space", groups=["ok", 7, "no good", None, "also.ok"])
    c = await idp.validator().validate(token)
    assert c.principal is None and c.groups == ("also.ok", "ok")
    c = await idp.validator().validate(idp.token(principal=["x"], groups="eng"))
    assert c.principal is None and c.groups == ()


def test_an_absurd_number_of_groups_is_capped() -> None:
    # (a real token this big is already refused by the validator's 8 KiB token-size limit)
    payload = claims(groups=[f"g{i}" for i in range(5000)])
    assert len(JwtTokenValidator._to_claims(payload).groups) == 500


async def test_an_oversized_groups_claim_never_reaches_parsing() -> None:
    idp = TestIdp()
    with pytest.raises(InvalidTokenError) as caught:
        await idp.validator().validate(idp.token(groups=[f"g{i}" for i in range(5000)]))
    assert caught.value.reason == "too_large"
