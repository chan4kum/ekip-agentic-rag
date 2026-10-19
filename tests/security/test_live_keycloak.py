"""The real app, wired to a REAL Keycloak, attacked with real (and doctored real) tokens.

Run: RUN_INTEGRATION=1 KEYCLOAK_HOST_PORT=8091 uv run pytest tests/security/test_live_keycloak.py
"""

import base64
import json
import os
from collections.abc import AsyncIterator

import httpx
import pytest
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_identity.jwks import CachedJwks, httpx_fetcher
from ekip_identity.validator import JwtTokenValidator
from idp_testkit import open_limits, tamper_payload

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
    async with httpx.AsyncClient(trust_env=False, timeout=10) as c:
        r = await c.post(
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
async def client() -> AsyncIterator[httpx.AsyncClient]:
    validator = JwtTokenValidator(
        issuer=ISSUER, audience="ekip-api", keys=CachedJwks(httpx_fetcher(JWKS_URL))
    )
    app = create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=validator,
        rate_limits=open_limits(),
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_real_tokens_work_and_are_tenant_scoped(client: httpx.AsyncClient) -> None:
    for user, prefix, other in (
        ("alice", "/tenant-dev/", "/tenant-other/"),
        ("bob", "/tenant-other/", "/tenant-dev/"),
    ):
        r = await client.get(
            "/api/v1/search",
            params={"q": "torque", "tenant_id": "tenant-dev"},
            headers={**_auth(await _token(user)), "X-Tenant-ID": "tenant-dev"},
        )
        assert r.status_code == 200
        ids = [x["source_uri"] for x in r.json()["results"]]
        assert ids and all(prefix in i for i in ids) and not any(other in i for i in ids)


async def test_real_token_on_chat_streams(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/api/v1/chat", json={"message": "torque"}, headers=_auth(await _token("alice"))
    )
    assert r.status_code == 200 and "event: token" in r.text


async def test_promoting_a_real_token_by_editing_claims_fails(client: httpx.AsyncClient) -> None:
    token = await _token("alice")
    for changes in ({"tenant_id": "tenant-other"}, {"roles": ["ekip-admin"]}):
        forged = tamper_payload(token, **changes)
        r = await client.get("/api/v1/search", params={"q": "x"}, headers=_auth(forged))
        assert r.status_code == 401, changes


async def test_alg_none_variant_of_a_real_token_fails(client: httpx.AsyncClient) -> None:
    token = await _token("alice")
    _, payload, _ = token.split(".")
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    unsigned = f"{header.rstrip(b'=').decode()}.{payload}."
    r = await client.get("/api/v1/search", params={"q": "x"}, headers=_auth(unsigned))
    assert r.status_code == 401


async def test_stripping_the_signature_fails(client: httpx.AsyncClient) -> None:
    head, payload, _ = (await _token("alice")).split(".")
    r = await client.get("/api/v1/search", params={"q": "x"}, headers=_auth(f"{head}.{payload}."))
    assert r.status_code == 401


async def test_a_user_role_cannot_use_admin_only_actions(client: httpx.AsyncClient) -> None:
    """Real roles are enforced: alice (ekip-user) holds search/read but not admin:manage."""
    from ekip_authorization.rbac import DEFAULT_POLICY
    from ekip_contracts.authorization import Action

    assert Action.ADMIN_MANAGE not in DEFAULT_POLICY["ekip-user"]
    assert Action.ADMIN_MANAGE in DEFAULT_POLICY["ekip-admin"]
