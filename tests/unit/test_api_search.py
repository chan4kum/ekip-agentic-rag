from collections.abc import AsyncIterator

import httpx
import pytest
from ekip_api.dependencies import get_retrieval_service
from ekip_api.main import create_app
from ekip_api.services.retrieval import RetrievalService
from ekip_config.settings import Settings
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from fastapi import FastAPI
from idp_testkit import TestIdp, open_limits

IDP = TestIdp()
ALICE = IDP.auth(tenant_id="tenant-dev")  # alice: group eng
BOB = IDP.auth(tenant_id="tenant-other", sub="bob", principal="bob")  # tenant-other
CAROL = IDP.auth(sub="carol", principal="carol", groups=["eng", "ops"])  # tenant-dev, also ops


def make_app(**kwargs: object) -> FastAPI:
    return create_app(
        Settings(_env_file=None, app_env="test"),
        token_validator=IDP.validator(),
        rate_limits=open_limits(),
        **kwargs,  # type: ignore[arg-type]
    )


def http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with http(make_app()) as c:
        yield c


class FixedRetriever:
    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk_id="fixed", document_id="d", text=query, score=1.0)]


async def test_search_returns_ranked_results(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/search", params={"q": "torque inspection"}, headers=ALICE)
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "torque inspection"
    assert body["results"][0]["document_id"]
    assert body["results"][0]["source_uri"].startswith("https://demo.test/tenant-dev/")


async def test_other_tenants_data_is_never_returned(client: httpx.AsyncClient) -> None:
    r = await client.get(
        "/api/v1/search", params={"q": "torque inspection", "limit": 20}, headers=ALICE
    )
    ids = [x["source_uri"] for x in r.json()["results"]]
    assert ids
    assert not any("/tenant-other/" in i for i in ids)


async def test_tenant_comes_from_the_token_for_each_caller(client: httpx.AsyncClient) -> None:
    alice = await client.get("/api/v1/search", params={"q": "torque", "limit": 20}, headers=ALICE)
    bob = await client.get("/api/v1/search", params={"q": "torque", "limit": 20}, headers=BOB)
    a_ids = {x["source_uri"] for x in alice.json()["results"]}
    b_ids = {x["source_uri"] for x in bob.json()["results"]}
    assert a_ids and b_ids
    assert all("/tenant-dev/" in i for i in a_ids)
    assert all("/tenant-other/" in i for i in b_ids)


async def test_tenant_cannot_be_chosen_by_the_caller(client: httpx.AsyncClient) -> None:
    """Alice asks for another tenant's data three different ways; the token still wins."""
    r = await client.get(
        "/api/v1/search",
        params={"q": "payroll", "tenant_id": "tenant-other"},
        headers={**ALICE, "X-Tenant-ID": "tenant-other", "X-Forwarded-Tenant": "tenant-other"},
    )
    assert r.status_code == 200
    # Vector search always returns its nearest neighbours, so weak hits from Alice's OWN tenant may
    # appear; what must never appear is anything of tenant-other (e.g. its payroll calendar).
    results = r.json()["results"]
    assert not any("/tenant-other/" in x["source_uri"] for x in results)
    assert not any("Payroll" in x["text"] for x in results)


@pytest.mark.parametrize(
    "params",
    [{}, {"q": "x", "limit": 0}, {"q": "x", "limit": 21}, {"q": "   "}],
)
async def test_invalid_requests_get_422(
    client: httpx.AsyncClient, params: dict[str, object]
) -> None:
    r = await client.get("/api/v1/search", params=params, headers=ALICE)  # type: ignore[arg-type]
    assert r.status_code == 422


async def test_retriever_can_be_injected_at_the_composition_root() -> None:
    async with http(make_app(retriever=FixedRetriever())) as c:
        r = await c.get("/api/v1/search", params={"q": "anything"}, headers=ALICE)
    assert r.json()["results"][0]["chunk_id"] == "fixed"


async def test_dependency_override_swaps_the_service() -> None:
    app = make_app()
    app.dependency_overrides[get_retrieval_service] = lambda: RetrievalService(FixedRetriever())
    async with http(app) as c:
        r = await c.get("/api/v1/search", params={"q": "anything"}, headers=ALICE)
    assert r.json()["results"][0]["chunk_id"] == "fixed"


# ---- document-level access inside one tenant -------------------------------------------------


def _titles(r: httpx.Response) -> set[str]:
    return {x["title"] for x in r.json()["results"]}


async def test_group_membership_decides_which_documents_come_back(
    client: httpx.AsyncClient,
) -> None:
    q: dict[str, str | int] = {"q": "torque inspection audit", "limit": 20}
    alice = await client.get("/api/v1/search", params=q, headers=ALICE)  # eng only
    carol = await client.get("/api/v1/search", params=q, headers=CAROL)  # eng + ops
    assert "Operations audit findings" not in _titles(alice)
    assert "Operations audit findings" in _titles(carol)
    assert _titles(alice) < _titles(
        carol
    )  # carol sees everything alice sees, plus the ops document


async def test_a_token_without_identity_claims_sees_nothing(client: httpx.AsyncClient) -> None:
    anonymous = IDP.auth(principal=None, groups=None)  # valid token, but no ACL identity
    r = await client.get("/api/v1/search", params={"q": "torque", "limit": 20}, headers=anonymous)
    assert r.status_code == 200 and r.json()["results"] == []  # fail closed


async def test_a_retriever_receives_the_scope_built_from_the_token() -> None:
    seen: list[SearchScope] = []

    class Spy:
        async def search(
            self, query: str, *, scope: SearchScope, limit: int = 5
        ) -> list[RetrievedChunk]:
            seen.append(scope)
            return []

    async with http(make_app(retriever=Spy())) as c:
        await c.get(
            "/api/v1/search",
            params={"q": "x", "tenant_id": "tenant-other", "user": "root"},
            headers=CAROL,
        )
    assert seen == [SearchScope("tenant-dev", user="carol", groups=frozenset({"eng", "ops"}))]
