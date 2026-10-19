import io
import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import structlog
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_contracts.identity import AccessTokenClaims, InvalidTokenError, TokenValidator
from ekip_observability.context import clear_request_context
from ekip_observability.logging import configure_logging
from fastapi import FastAPI
from idp_testkit import AUDIENCE, ISSUER, TestIdp, make_key, mint, open_limits

IDP = TestIdp()
URL = "/api/v1/search"


def app_with(validator: TokenValidator) -> FastAPI:
    return create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=validator,
        rate_limits=open_limits(),
    )


def http(app: FastAPI, *, raise_exceptions: bool = True) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_exceptions)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with http(app_with(IDP.validator())) as c:
        yield c


@pytest.fixture
def log_buf() -> Iterator[io.StringIO]:
    structlog.reset_defaults()
    clear_request_context()
    stream = io.StringIO()
    configure_logging("INFO", json_logs=True, stream=stream)
    yield stream
    clear_request_context()
    structlog.reset_defaults()


async def test_no_authorization_header_is_401_with_bare_challenge(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get(URL, params={"q": "torque"})
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("header", ["Basic dXNlcjpwYXNz", "Bearer", "Bearer ", "Token abc", "abc"])
async def test_wrong_scheme_or_empty_credentials_is_401(
    client: httpx.AsyncClient, header: str
) -> None:
    r = await client.get(URL, params={"q": "torque"}, headers={"Authorization": header})
    assert r.status_code == 401


async def test_token_in_query_string_is_not_accepted(client: httpx.AsyncClient) -> None:
    r = await client.get(URL, params={"q": "torque", "access_token": IDP.token()})
    assert r.status_code == 401


def _bad_tokens() -> dict[str, str]:
    import time

    return {
        "garbage": "not-a-jwt",
        "expired": IDP.token(exp=1_000_000_000, iat=999_999_000),
        "wrong_audience": IDP.token(aud="someone-else"),
        "wrong_issuer": IDP.token(iss="http://evil.test/realms/ekip"),
        "foreign_key": mint(make_key(), kid=IDP.kid),
        "unknown_kid": mint(IDP.key, kid="nope"),
        "no_tenant": IDP.token(tenant_id=None),
        "not_yet_valid": IDP.token(nbf=int(time.time()) + 3600),
    }


@pytest.mark.parametrize("name", list(_bad_tokens()))
async def test_every_invalid_token_gets_the_same_generic_401(
    client: httpx.AsyncClient, name: str
) -> None:
    r = await client.get(
        URL, params={"q": "torque"}, headers={"Authorization": f"Bearer {_bad_tokens()[name]}"}
    )
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == 'Bearer error="invalid_token"'
    # identical problem for every failure (only the request id differs): no oracle for attackers
    body = r.json()
    assert body.pop("request_id") == r.headers["x-request-id"]
    assert body == {
        "type": "urn:ekip:problem:unauthorized",
        "title": "Unauthorized",
        "status": 401,
        "detail": "authentication required",
    }


async def test_identity_provider_outage_is_503_not_401() -> None:
    class Down:
        async def validate(self, token: str) -> AccessTokenClaims:
            raise InvalidTokenError("key_unavailable")

    async with http(app_with(Down())) as c:
        r = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert r.status_code == 503
    assert "www-authenticate" not in r.headers


async def test_unexpected_validator_error_fails_closed() -> None:
    class Broken:
        async def validate(self, token: str) -> AccessTokenClaims:
            raise RuntimeError("bug")

    async with http(app_with(Broken()), raise_exceptions=False) as c:
        r = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert r.status_code == 500  # never an implicit allow


async def test_health_and_ready_need_no_token(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).status_code == 200


async def test_default_validator_is_wired_from_settings_and_denies_unknown_tokens() -> None:
    """No injected validator: the real JwtTokenValidator is built from settings."""
    app = create_app(Settings(_env_file=None, app_env="test"))
    async with http(app) as c:
        r = await c.get(URL, params={"q": "x"}, headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401
    assert AUDIENCE == "ekip-api" and ISSUER.startswith("http")  # keep helper constants honest


async def test_log_line_has_tenant_and_subject_but_never_the_token(
    client: httpx.AsyncClient, log_buf: io.StringIO
) -> None:
    token = IDP.token(tenant_id="tenant-dev", sub="user-42")
    await client.get(URL, params={"q": "torque"}, headers={"Authorization": f"Bearer {token}"})
    out = log_buf.getvalue()
    assert token not in out
    assert "Bearer" not in out
    access = [json.loads(line) for line in out.splitlines() if '"http_request"' in line]
    assert access[-1]["tenant_id"] == "tenant-dev"
    assert access[-1]["subject"] == "user-42"


async def test_rejected_request_logs_reason_without_the_token(
    client: httpx.AsyncClient, log_buf: io.StringIO
) -> None:
    token = IDP.token(aud="someone-else")
    await client.get(URL, params={"q": "torque"}, headers={"Authorization": f"Bearer {token}"})
    out = log_buf.getvalue()
    assert "wrong_audience" in out
    assert token not in out
