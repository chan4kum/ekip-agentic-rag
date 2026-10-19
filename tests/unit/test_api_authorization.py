import io
import json
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
import structlog
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_contracts.authorization import Action, Decision, Resource
from ekip_contracts.identity import SecurityContext
from ekip_observability.context import clear_request_context
from ekip_observability.logging import configure_logging
from fastapi import FastAPI
from idp_testkit import TestIdp, open_limits

IDP = TestIdp()
URL = "/api/v1/search"


def app_with(**kwargs: object) -> FastAPI:
    return create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
        **kwargs,  # type: ignore[arg-type]
    )


def http(app: FastAPI, *, raise_exceptions: bool = True) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_exceptions)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with http(app_with()) as c:
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


async def test_user_role_may_search(client: httpx.AsyncClient) -> None:
    r = await client.get(URL, params={"q": "torque"}, headers=IDP.auth(roles=["ekip-user"]))
    assert r.status_code == 200


async def test_admin_role_may_search(client: httpx.AsyncClient) -> None:
    r = await client.get(URL, params={"q": "torque"}, headers=IDP.auth(roles=["ekip-admin"]))
    assert r.status_code == 200


@pytest.mark.parametrize("roles", [[], ["hacker"], ["ekip-user-readonly"], ["EKIP-USER"]])
async def test_valid_token_without_a_granting_role_is_403(
    client: httpx.AsyncClient, roles: list[str]
) -> None:
    r = await client.get(URL, params={"q": "torque"}, headers=IDP.auth(roles=roles))
    assert r.status_code == 403
    assert r.json()["detail"] == "forbidden"  # no policy detail leaks to the client
    assert set(r.json()) == {"type", "title", "status", "detail", "request_id"}
    assert r.headers["cache-control"] == "no-store"


async def test_token_without_roles_claim_is_403(client: httpx.AsyncClient) -> None:
    r = await client.get(URL, params={"q": "torque"}, headers=IDP.auth(roles=None))
    assert r.status_code == 403


async def test_authentication_failure_wins_over_authorization(client: httpx.AsyncClient) -> None:
    assert (await client.get(URL, params={"q": "torque"})).status_code == 401


async def test_denial_is_logged_with_reason_action_and_caller(
    client: httpx.AsyncClient, log_buf: io.StringIO
) -> None:
    await client.get(URL, params={"q": "torque"}, headers=IDP.auth(roles=["hacker"], sub="eve"))
    lines = [json.loads(x) for x in log_buf.getvalue().splitlines() if x.strip()]
    denied = next(x for x in lines if x["event"] == "authorization_denied")
    assert denied["reason"] == "no_role_grants_action"
    assert denied["action"] == "documents:search"
    assert denied["tenant_id"] == "tenant-dev" and denied["subject"] == "eve"


async def test_injected_authorizer_is_used_and_receives_the_right_inputs() -> None:
    seen: list[tuple[SecurityContext, Action, Resource]] = []

    class Spy:
        async def authorize(
            self, context: SecurityContext, action: Action, resource: Resource
        ) -> Decision:
            seen.append((context, action, resource))
            return Decision(allowed=True, reason="allowed")

    async with http(app_with(authorizer=Spy())) as c:
        r = await c.get(
            URL, params={"q": "x"}, headers=IDP.auth(roles=[], tenant_id="tenant-other")
        )
    assert r.status_code == 200  # the injected policy decides, not the token's role list
    ((context, action, resource),) = seen
    assert action is Action.DOCUMENTS_SEARCH
    assert context.tenant_id == "tenant-other"
    assert resource.tenant_id == "tenant-other"  # derived from the token, not from the request


async def test_deny_all_authorizer_blocks_everything() -> None:
    class DenyAll:
        async def authorize(
            self, context: SecurityContext, action: Action, resource: Resource
        ) -> Decision:
            return Decision(allowed=False, reason="no_role_grants_action")

    async with http(app_with(authorizer=DenyAll())) as c:
        r = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert r.status_code == 403


async def test_authorizer_outage_is_503_not_an_allow() -> None:
    class Down:
        async def authorize(
            self, context: SecurityContext, action: Action, resource: Resource
        ) -> Decision:
            raise ConnectionError("policy service unreachable")

    async with http(app_with(authorizer=Down())) as c:
        r = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert r.status_code == 503
