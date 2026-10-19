"""Threat-organised security regression suite for the API boundary (Phase 2).

Every protected route is attacked with every forged / defective credential. Adding a route that is
not classified below fails `test_every_route_is_classified`: a deliberate security decision.
"""

import io
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx
import pytest
import structlog
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_observability.context import clear_request_context
from ekip_observability.logging import configure_logging
from fastapi import FastAPI
from fastapi.routing import APIRoute
from idp_testkit import (
    TestIdp,
    forge_alg_none,
    forge_hs256,
    make_key,
    mint,
    open_limits,
    tamper_payload,
)

IDP = TestIdp()
FOREIGN = make_key()

PUBLIC = {"/health", "/ready"}
PROTECTED: dict[tuple[str, str], dict[str, Any]] = {
    ("GET", "/api/v1/search"): {"params": {"q": "torque"}},
    ("POST", "/api/v1/chat"): {"json": {"message": "torque"}},
}


def build_app() -> FastAPI:
    return create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
    )


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=build_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
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


def send(c: httpx.AsyncClient, route: tuple[str, str], token: str | None, **extra: Any) -> Any:
    method, path = route
    headers = dict(extra.pop("headers", {}))
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return c.request(method, path, headers=headers, **{**PROTECTED[route], **extra})


# ---- the attack catalogue: name -> token factory ------------------------------------------------
def _now() -> int:
    return int(time.time())


ATTACKS: dict[str, Callable[[], str]] = {
    "expired": lambda: IDP.token(exp=_now() - 3600, iat=_now() - 7200),
    "not_yet_valid": lambda: IDP.token(nbf=_now() + 3600),
    "wrong_audience": lambda: IDP.token(aud="some-other-api"),
    "audience_list_without_ours": lambda: IDP.token(aud=["account", "other"]),
    "wrong_issuer": lambda: IDP.token(iss="http://evil.test/realms/ekip"),
    "alg_none": lambda: forge_alg_none(),
    "alg_none_admin": lambda: forge_alg_none(roles=["ekip-admin"]),
    "hs256_key_confusion": lambda: forge_hs256(IDP.key),
    "foreign_signing_key": lambda: mint(FOREIGN, kid=IDP.kid),
    "foreign_key_claiming_admin": lambda: mint(FOREIGN, kid=IDP.kid, roles=["ekip-admin"]),
    "unknown_kid": lambda: mint(IDP.key, kid="rotated-away"),
    "tampered_tenant": lambda: tamper_payload(IDP.token(), tenant_id="tenant-other"),
    "tampered_roles": lambda: tamper_payload(IDP.token(roles=[]), roles=["ekip-admin"]),
    "no_tenant": lambda: IDP.token(tenant_id=None),
    "tenant_with_path_chars": lambda: IDP.token(tenant_id="../tenant-other"),
    "missing_subject": lambda: IDP.token(sub=None),
    "token_in_jku_header": lambda: mint(
        FOREIGN, kid=IDP.kid, headers={"jku": "http://evil.test/jwks.json"}
    ),
    "garbage": lambda: "not.a.jwt",
    "empty_segments": lambda: "..",
    "oversized": lambda: "a" * 20_000,
}


def _declared_operations(app: FastAPI) -> set[tuple[str, str]]:
    """(METHOD, path) pairs from the OpenAPI schema; independent of FastAPI's internal routing."""
    paths = app.openapi()["paths"]
    return {(m.upper(), p) for p, ops in paths.items() for m in ops}


def _hidden_routes(app: FastAPI) -> list[str]:
    """Routes excluded from the schema would escape the inventory; there must be none."""
    hidden: list[str] = []
    for route in app.routes:
        original = getattr(route, "original_router", None)
        for inner in getattr(original, "routes", []):
            if isinstance(inner, APIRoute) and not inner.include_in_schema:
                hidden.append(inner.path)
    return hidden


def test_every_route_is_classified() -> None:
    """A new route must be added to PUBLIC or PROTECTED (and so to every test below)."""
    app = build_app()
    seen = _declared_operations(app)
    classified = set(PROTECTED) | {("GET", p) for p in PUBLIC}
    unclassified = seen - classified
    assert not unclassified, f"classify these routes as public or protected: {sorted(unclassified)}"
    assert classified <= seen, f"classified but not served: {sorted(classified - seen)}"
    assert not _hidden_routes(app), "routes hidden from OpenAPI escape the security inventory"


@pytest.mark.parametrize("route", list(PROTECTED), ids=lambda r: f"{r[0]} {r[1]}")
async def test_no_credentials_is_401_before_validation(
    client: httpx.AsyncClient, route: tuple[str, str]
) -> None:
    r = await send(client, route, None)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    # even a malformed body must not be probed by an unauthenticated caller (401, not 422)
    r = await send(client, route, None, params={}, json=None, content=b"{not json")
    assert r.status_code == 401


@pytest.mark.parametrize("attack", list(ATTACKS))
@pytest.mark.parametrize("route", list(PROTECTED), ids=lambda r: f"{r[0]} {r[1]}")
async def test_every_attack_is_a_uniform_401_on_every_protected_route(
    client: httpx.AsyncClient, route: tuple[str, str], attack: str
) -> None:
    r = await send(client, route, ATTACKS[attack]())
    assert r.status_code == 401, attack
    body = r.json()
    body.pop("request_id")
    assert body == {
        "type": "urn:ekip:problem:unauthorized",
        "title": "Unauthorized",
        "status": 401,
        "detail": "authentication required",
    }  # identical for every attack: nothing to learn about which check caught it
    assert r.headers["www-authenticate"] == 'Bearer error="invalid_token"'
    assert "traceback" not in r.text.lower()


@pytest.mark.parametrize("route", list(PROTECTED), ids=lambda r: f"{r[0]} {r[1]}")
async def test_non_bearer_schemes_and_token_in_url_are_ignored(
    client: httpx.AsyncClient, route: tuple[str, str]
) -> None:
    good = IDP.token()
    for headers in (
        {"Authorization": f"Basic {good}"},
        {"Authorization": f"Token {good}"},
        {"Authorization": good},
        {"X-Access-Token": good},
        {"Cookie": f"access_token={good}"},
    ):
        r = await send(client, route, None, headers=headers)
        assert r.status_code == 401, headers
    r = await send(
        client, route, None, params={**PROTECTED[route].get("params", {}), "access_token": good}
    )
    assert r.status_code == 401


async def test_expiry_leeway_is_small_and_bounded(client: httpx.AsyncClient) -> None:
    route = ("GET", "/api/v1/search")
    just_expired = IDP.token(exp=_now() - 5, iat=_now() - 400)  # inside the 10 s clock-skew leeway
    long_expired = IDP.token(exp=_now() - 30, iat=_now() - 400)
    assert (await send(client, route, just_expired)).status_code == 200
    assert (await send(client, route, long_expired)).status_code == 401


# ---- authorization & tenant isolation -----------------------------------------------------------
@pytest.mark.parametrize("route", list(PROTECTED), ids=lambda r: f"{r[0]} {r[1]}")
@pytest.mark.parametrize("roles", [[], ["hacker"], ["EKIP-ADMIN"], ["ekip-admin "], None])
async def test_valid_token_without_a_granting_role_is_403(
    client: httpx.AsyncClient, route: tuple[str, str], roles: list[str] | None
) -> None:
    r = await send(client, route, IDP.token(roles=roles))
    assert r.status_code == 403
    assert r.json()["detail"] == "forbidden"


@pytest.mark.parametrize("route", list(PROTECTED), ids=lambda r: f"{r[0]} {r[1]}")
async def test_identity_headers_cannot_grant_roles_or_change_tenant(
    client: httpx.AsyncClient, route: tuple[str, str]
) -> None:
    spoof = {
        "X-Tenant-ID": "tenant-other",
        "X-User-ID": "admin",
        "X-Forwarded-User": "admin",
        "X-Roles": "ekip-admin",
        "X-Forwarded-Tenant": "tenant-other",
    }
    assert (await send(client, route, None, headers=spoof)).status_code == 401
    assert (await send(client, route, IDP.token(roles=[]), headers=spoof)).status_code == 403


def _ids_from(response: httpx.Response, route: tuple[str, str]) -> set[str]:
    if route[1].endswith("search"):
        return {x["source_uri"] for x in response.json()["results"]}
    ids: set[str] = set()
    for block in response.text.split("\n\n"):
        lines = block.split("\n")
        if lines[0] == "event: citations":
            ids |= {x["source_uri"] for x in json.loads(lines[1][6:])["results"]}
    return ids


@pytest.mark.parametrize("route", list(PROTECTED), ids=lambda r: f"{r[0]} {r[1]}")
@pytest.mark.parametrize(
    ("tenant", "prefix", "foreign_prefix"),
    [
        ("tenant-dev", "/tenant-dev/", "/tenant-other/"),
        ("tenant-other", "/tenant-other/", "/tenant-dev/"),
    ],
)
async def test_tenants_never_see_each_others_data_even_when_they_ask(
    client: httpx.AsyncClient,
    route: tuple[str, str],
    tenant: str,
    prefix: str,
    foreign_prefix: str,
) -> None:
    other = "tenant-other" if tenant == "tenant-dev" else "tenant-dev"
    headers = {"X-Tenant-ID": other, "X-Forwarded-Tenant": other}
    extra: dict[str, Any] = {"headers": headers}
    if route[0] == "GET":
        extra["params"] = {"q": "torque", "limit": 20, "tenant_id": other, "tenant": other}
    ids = _ids_from(await send(client, route, IDP.token(tenant_id=tenant), **extra), route)
    assert ids, "the caller's own data should be found"
    assert all(prefix in i for i in ids)
    assert not any(foreign_prefix in i for i in ids)


async def test_chat_rejects_a_tenant_in_the_body(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/api/v1/chat",
        json={"message": "torque", "tenant_id": "tenant-other"},
        headers=IDP.auth(),
    )
    assert r.status_code == 422


# ---- confidentiality of credentials -------------------------------------------------------------
async def test_no_token_material_ever_reaches_the_logs(
    client: httpx.AsyncClient, log_buf: io.StringIO
) -> None:
    tokens = [make() for make in ATTACKS.values() if make is not ATTACKS["oversized"]]
    tokens.append(IDP.token())
    for token in tokens:
        for route in PROTECTED:
            await send(client, route, token)
    logs = log_buf.getvalue()
    assert logs  # we did capture something
    for token in tokens:
        for segment in token.split("."):
            if len(segment) >= 20:  # long enough to be a real header/payload/signature
                assert segment not in logs, "a token segment leaked into the logs"
    assert "Bearer " not in logs


# ---- production wiring --------------------------------------------------------------------------
async def test_production_defaults_reject_a_token_from_an_untrusted_idp() -> None:
    """No injected validator: the real one, pointed at an unreachable IdP, must fail closed."""
    settings = Settings(
        _env_file=None,
        app_env="production",
        database_url="postgresql+asyncpg://u:p@db.internal:5432/db",
        redis_url="redis://cache.internal:6379/0",
        opensearch_url="https://search.internal:9200",
        oidc_issuer="https://idp.invalid/realms/ekip",
        oidc_jwks_url="https://idp.invalid/realms/ekip/protocol/openid-connect/certs",
    )
    app = create_app(settings, rate_limits=open_limits())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/v1/search", params={"q": "x"}, headers=IDP.auth())
        docs = await c.get("/docs")
        schema = await c.get("/openapi.json")
    assert r.status_code in {401, 503}  # never 200: unknown IdP => no trust
    assert docs.status_code == 404 and schema.status_code == 404
