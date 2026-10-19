"""Validate REAL Keycloak tokens.

Run: RUN_INTEGRATION=1 KEYCLOAK_HOST_PORT=8091 uv run pytest tests/integration
"""

import os

import httpx
import jwt
import pytest
from ekip_contracts.identity import InvalidTokenError
from ekip_identity.jwks import CachedJwks, httpx_fetcher
from ekip_identity.validator import JwtTokenValidator

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs live Compose stack"),
]

BASE = (
    os.environ.get("KEYCLOAK_URL")
    or f"http://localhost:{os.environ.get('KEYCLOAK_HOST_PORT', '8081')}"
)
ISSUER = f"{BASE}/realms/ekip"
JWKS_URL = f"{ISSUER}/protocol/openid-connect/certs"


async def _token(user: str) -> str:
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        r = await client.post(
            f"{ISSUER}/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "ekip-test",
                "username": user,
                "password": f"{user}-dev-password",
                "scope": "openid",
            },
        )
    r.raise_for_status()
    token: str = r.json()["access_token"]
    return token


@pytest.fixture
def validator() -> JwtTokenValidator:
    return JwtTokenValidator(
        issuer=ISSUER,
        audience="ekip-api",
        keys=CachedJwks(httpx_fetcher(JWKS_URL)),
    )


async def test_alice_token_is_valid_with_her_tenant(validator: JwtTokenValidator) -> None:
    c = await validator.validate(await _token("alice"))
    assert c.tenant_id == "tenant-dev"
    assert "ekip-user" in c.roles
    assert "ekip-api" in c.audience


async def test_bob_is_in_a_different_tenant(validator: JwtTokenValidator) -> None:
    assert (await validator.validate(await _token("bob"))).tenant_id == "tenant-other"


async def test_carol_has_admin_role(validator: JwtTokenValidator) -> None:
    assert "ekip-admin" in (await validator.validate(await _token("carol"))).roles


async def test_tampered_real_token_is_rejected(validator: JwtTokenValidator) -> None:
    token = await _token("alice")
    head, payload, sig = token.split(".")
    flipped = payload[:-2] + ("AA" if not payload.endswith("AA") else "BB")
    with pytest.raises(InvalidTokenError) as exc:
        await validator.validate(f"{head}.{flipped}.{sig}")
    assert exc.value.reason in {"bad_signature", "malformed"}


async def test_audience_mismatch_with_real_token() -> None:
    v = JwtTokenValidator(
        issuer=ISSUER, audience="some-other-api", keys=CachedJwks(httpx_fetcher(JWKS_URL))
    )
    with pytest.raises(InvalidTokenError) as exc:
        await v.validate(await _token("alice"))
    assert exc.value.reason == "wrong_audience"


async def test_wrong_configured_issuer_rejects_real_token() -> None:
    v = JwtTokenValidator(
        issuer="http://evil.test/realms/ekip",
        audience="ekip-api",
        keys=CachedJwks(httpx_fetcher(JWKS_URL)),
    )
    with pytest.raises(InvalidTokenError) as exc:
        await v.validate(await _token("alice"))
    assert exc.value.reason == "wrong_issuer"


async def test_real_token_header_uses_allowed_algorithm_and_kid() -> None:
    header = jwt.get_unverified_header(await _token("alice"))
    assert header["alg"] == "RS256"
    assert isinstance(header["kid"], str)
