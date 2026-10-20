import json

import httpx
import pytest
from ekip_contracts.identity import SecurityContext
from ekip_tools import OutboundBlockedError, TicketBackend, ToolBackendError, WebSearchBackend
from tools_testkit import ALICE, make_spec, write_spec

HOST = "tickets.example.com"


class Creds:
    def __init__(self) -> None:
        self.audiences: list[str] = []

    async def token_for(self, context: SecurityContext, audience: str) -> str:
        self.audiences.append(audience)
        return f"exchanged-for-{audience}"


async def public(_: str) -> list[str]:
    return ["93.184.216.34"]


async def internal(_: str) -> list[str]:
    return ["10.0.0.9"]


def client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


def ticket_backend(handler, *, resolver=public, projects=None) -> tuple[TicketBackend, Creds]:  # type: ignore[no-untyped-def]
    creds = Creds()
    backend = TicketBackend(
        client(httpx.MockTransport(handler)),
        base_url=f"https://{HOST}/api",
        allowed_hosts=[HOST],
        project_for_tenant=projects or {"tenant-a": "PROJ-A"},
        credentials=creds,
        resolver=resolver,
    )
    return backend, creds


async def test_ticket_is_created_in_the_callers_tenant_project() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(201, json={"key": "PROJ-A-42"})

    backend, creds = ticket_backend(handler)
    out = await backend.call(ALICE, write_spec(), {"title": "Broken pump", "priority": "high"})
    request = seen["request"]
    assert out == "created ticket PROJ-A-42"
    assert request.url.path == "/api/projects/PROJ-A/issues"
    assert request.headers["authorization"] == f"Bearer exchanged-for-{HOST}"
    assert json.loads(request.content)["reporter"] == "u-alice"
    assert creds.audiences == [HOST]


async def test_unconfigured_tenant_and_internal_destination_are_refused() -> None:
    backend, _ = ticket_backend(lambda r: httpx.Response(201, json={}), projects={"other": "X"})
    with pytest.raises(ToolBackendError, match="not configured"):
        await backend.call(ALICE, write_spec(), {"title": "abc"})
    backend, _ = ticket_backend(lambda r: httpx.Response(201, json={}), resolver=internal)
    with pytest.raises(ToolBackendError, match="destination not allowed"):
        await backend.call(ALICE, write_spec(), {"title": "abc"})


async def test_ticket_failures_are_never_retryable() -> None:
    backend, _ = ticket_backend(lambda r: httpx.Response(503))
    with pytest.raises(ToolBackendError) as err:
        await backend.call(ALICE, write_spec(), {"title": "abc"})
    assert err.value.retryable is False
    backend, _ = ticket_backend(
        lambda r: httpx.Response(302, headers={"location": "http://169.254.169.254/"})
    )
    with pytest.raises(ToolBackendError, match="redirect"):
        await backend.call(ALICE, write_spec(), {"title": "abc"})


def test_backends_refuse_non_allowlisted_endpoints_at_construction() -> None:
    with pytest.raises(OutboundBlockedError):
        TicketBackend(
            client(httpx.MockTransport(lambda r: httpx.Response(200))),
            base_url="https://evil.example.org/api",
            allowed_hosts=[HOST],
            project_for_tenant={},
            credentials=Creds(),
        )


def search_backend(handler, **kw) -> WebSearchBackend:  # type: ignore[no-untyped-def]
    return WebSearchBackend(
        client(httpx.MockTransport(handler)),
        endpoint="https://search.example.com/v1/search",
        allowed_hosts=["search.example.com"],
        credentials=Creds(),
        resolver=kw.pop("resolver", public),
        **kw,
    )


SEARCH = make_spec("web_search", idempotent=True, max_retries=1)


async def test_web_search_formats_results_and_passes_the_query() -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200, json={"results": [{"title": "T", "url": "https://a.example/x", "snippet": "S"}]}
        )

    out = await search_backend(handler).call(ALICE, SEARCH, {"query": "torque values", "count": 3})
    assert out == "- T (https://a.example/x): S"
    assert seen["request"].url.params["q"] == "torque values"
    assert seen["request"].url.params["count"] == "3"


async def test_web_search_query_filter_can_rewrite_or_block() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    def block(query: str) -> str:
        if "secret" in query:
            raise ToolBackendError("query blocked by policy")
        return query

    backend = search_backend(handler, query_filter=block)
    with pytest.raises(ToolBackendError, match="blocked"):
        await backend.call(ALICE, SEARCH, {"query": "our secret plan"})
    assert await backend.call(ALICE, SEARCH, {"query": "weather"}) == "no results"


async def test_web_search_bounds_the_response_and_checks_shape() -> None:
    big = search_backend(lambda r: httpx.Response(200, content=b"x" * (300 * 1024)))
    with pytest.raises(ToolBackendError, match="too large"):
        await big.call(ALICE, SEARCH, {"query": "q"})
    odd = search_backend(lambda r: httpx.Response(200, json={"nope": 1}))
    with pytest.raises(ToolBackendError, match="unexpected shape"):
        await odd.call(ALICE, SEARCH, {"query": "q"})
    junk = search_backend(lambda r: httpx.Response(200, content=b"<html>"))
    with pytest.raises(ToolBackendError, match="invalid JSON"):
        await junk.call(ALICE, SEARCH, {"query": "q"})


async def test_web_search_retry_flag_follows_idempotency() -> None:
    backend = search_backend(lambda r: httpx.Response(503))
    with pytest.raises(ToolBackendError) as err:
        await backend.call(ALICE, SEARCH, {"query": "q"})
    assert err.value.retryable is True


async def test_web_search_blocks_internal_resolution() -> None:
    backend = search_backend(lambda r: httpx.Response(200, json={"results": []}), resolver=internal)
    with pytest.raises(ToolBackendError, match="destination not allowed"):
        await backend.call(ALICE, SEARCH, {"query": "q"})
