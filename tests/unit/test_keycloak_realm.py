"""Guards for the dev identity provider: realm file and Compose service (no Docker needed)."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def realm() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (ROOT / "infrastructure/keycloak/realm-ekip.json").read_text()
    )
    return data


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    return data


def client(realm: dict[str, Any], client_id: str) -> dict[str, Any]:
    return next(c for c in realm["clients"] if c["clientId"] == client_id)


def test_realm_is_clearly_marked_as_development_only(realm: dict[str, Any]) -> None:
    assert realm["realm"] == "ekip"
    assert "DEVELOPMENT" in realm["displayName"]


def test_self_registration_is_off_and_brute_force_protection_is_on(realm: dict[str, Any]) -> None:
    assert realm["registrationAllowed"] is False
    assert realm["bruteForceProtected"] is True


def test_access_tokens_are_short_lived(realm: dict[str, Any]) -> None:
    assert realm["accessTokenLifespan"] <= 600


def test_web_client_is_public_with_mandatory_pkce_and_no_risky_flows(realm: dict[str, Any]) -> None:
    web = client(realm, "ekip-web")
    assert web["publicClient"] is True and "secret" not in web
    assert web["standardFlowEnabled"] is True
    assert web["attributes"]["pkce.code.challenge.method"] == "S256"
    assert web["implicitFlowEnabled"] is False
    assert web["directAccessGrantsEnabled"] is False


def test_redirect_uris_and_origins_are_exact_never_wildcard_hosts(realm: dict[str, Any]) -> None:
    web = client(realm, "ekip-web")
    for uri in web["redirectUris"]:
        assert uri != "*" and re.match(r"^https?://[a-z0-9.:-]+/", uri)
    assert "*" not in web["webOrigins"]


def test_api_client_is_bearer_only_and_logs_nobody_in(realm: dict[str, Any]) -> None:
    api = client(realm, "ekip-api")
    assert api["bearerOnly"] is True
    assert not any(api.get(f) for f in ("standardFlowEnabled", "directAccessGrantsEnabled"))


def test_only_the_dev_test_client_may_use_the_password_grant(realm: dict[str, Any]) -> None:
    with_password_grant = [
        c["clientId"] for c in realm["clients"] if c["directAccessGrantsEnabled"]
    ]
    assert with_password_grant == ["ekip-test"]
    assert client(realm, "ekip-test")["attributes"]["ekip.dev-only"] == "true"


@pytest.mark.parametrize("client_id", ["ekip-web", "ekip-test"])
def test_tokens_carry_audience_tenant_and_roles(realm: dict[str, Any], client_id: str) -> None:
    mappers = {m["name"]: m for m in client(realm, client_id)["protocolMappers"]}
    assert mappers["ekip-api-audience"]["config"]["included.custom.audience"] == "ekip-api"
    assert mappers["tenant-id"]["config"]["claim.name"] == "tenant_id"
    assert mappers["tenant-id"]["config"]["user.attribute"] == "tenant_id"
    assert mappers["flat-roles"]["config"]["claim.name"] == "roles"
    for m in mappers.values():
        assert m["config"]["access.token.claim"] == "true"
        assert m["config"]["id.token.claim"] == "false"  # claims are for the API, not the ID token


def test_users_cannot_edit_their_own_tenant(realm: dict[str, Any]) -> None:
    components = realm["components"]["org.keycloak.userprofile.UserProfileProvider"]
    profile = json.loads(components[0]["config"]["kc.user.profile.config"][0])
    tenant = next(a for a in profile["attributes"] if a["name"] == "tenant_id")
    assert tenant["permissions"] == {"view": ["admin"], "edit": ["admin"]}


def test_every_dev_user_belongs_to_a_tenant_and_there_are_two_tenants(
    realm: dict[str, Any],
) -> None:
    tenants = {u["username"]: u["attributes"]["tenant_id"][0] for u in realm["users"]}
    assert set(tenants) == {"alice", "bob", "carol"}
    assert len(set(tenants.values())) == 2  # cross-tenant tests need two real tenants
    assert all(not u["credentials"][0]["temporary"] for u in realm["users"])


def test_roles_exist_and_users_only_use_defined_roles(realm: dict[str, Any]) -> None:
    defined = {r["name"] for r in realm["roles"]["realm"]}
    assert defined == {"ekip-user", "ekip-admin"}
    for u in realm["users"]:
        assert set(u["realmRoles"]) <= defined


def test_keycloak_service_is_localhost_only_digest_pinned_and_locked_down(
    compose: dict[str, Any],
) -> None:
    kc = compose["services"]["keycloak"]
    assert kc["ports"] == ["127.0.0.1:${KEYCLOAK_HOST_PORT:-8081}:8080"]
    assert re.search(r"@sha256:[0-9a-f]{64}$", kc["image"])
    assert kc["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in kc["security_opt"]
    assert "egress" not in kc["networks"]  # no internet access


def test_keycloak_issuer_is_fixed_but_internal_key_fetch_still_works(
    compose: dict[str, Any],
) -> None:
    env = compose["services"]["keycloak"]["environment"]
    # the issuer must follow the published port, otherwise tokens carry an unreachable iss
    assert env["KC_HOSTNAME"] == "http://localhost:${KEYCLOAK_HOST_PORT:-8081}"
    assert env["KC_HOSTNAME_BACKCHANNEL_DYNAMIC"] == "true"


def test_realm_import_file_is_mounted_read_only(compose: dict[str, Any]) -> None:
    volumes = compose["services"]["keycloak"]["volumes"]
    assert any("realm-ekip.json" in v and v.endswith(":ro") for v in volumes)


def test_keycloak_has_a_healthcheck_and_is_dev_mode(compose: dict[str, Any]) -> None:
    kc = compose["services"]["keycloak"]
    assert "healthcheck" in kc
    assert kc["command"][0] == "start-dev"  # dev-only service; production uses a built image


@pytest.mark.parametrize("client_id", ["ekip-web", "ekip-test"])
def test_tokens_carry_the_acl_identity(realm: dict[str, Any], client_id: str) -> None:
    mappers = {m["name"]: m for m in client(realm, client_id)["protocolMappers"]}
    principal, groups = mappers["principal"]["config"], mappers["groups"]["config"]
    assert (principal["user.attribute"], principal["claim.name"]) == ("username", "principal")
    assert groups["claim.name"] == "groups"
    assert groups["full.path"] == "false"  # "eng", not "/eng": matches ACL group names
    assert mappers["groups"]["protocolMapper"] == "oidc-group-membership-mapper"


def test_dev_users_have_the_expected_groups(realm: dict[str, Any]) -> None:
    assert {g["name"] for g in realm["groups"]} == {"eng", "ops"}
    members = {u["username"]: set(u["groups"]) for u in realm["users"]}
    assert members == {"alice": {"/eng"}, "bob": {"/eng"}, "carol": {"/eng", "/ops"}}


def test_principal_comes_from_a_username_nobody_can_edit(realm: dict[str, Any]) -> None:
    assert realm["editUsernameAllowed"] is False  # the ACL identity cannot be self-renamed
