from collections.abc import AsyncIterator

import httpx
import pytest
from ekip_api.dependencies import get_retrieval_service
from ekip_api.main import create_app
from ekip_api.services.retrieval import RetrievalService
from ekip_config.settings import Settings
from ekip_contracts.retrieval import RetrievedChunk
from fastapi import FastAPI


def make_app(**kwargs: object) -> FastAPI:
    return create_app(Settings(_env_file=None, app_env="test"), **kwargs)  # type: ignore[arg-type]


def http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with http(make_app()) as c:
        yield c


class FixedRetriever:
    async def search(self, query: str, *, tenant_id: str, limit: int = 5) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk_id="fixed", document_id="d", text=query, score=1.0)]


async def test_search_returns_ranked_results(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/search", params={"q": "torque inspection"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "torque inspection"
    assert body["results"][0]["chunk_id"] in {"dev-1", "dev-2"}


async def test_other_tenants_data_is_never_returned(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/search", params={"q": "torque inspection", "limit": 20})
    ids = [x["chunk_id"] for x in r.json()["results"]]
    assert ids
    assert not any(i.startswith("oth-") for i in ids)


async def test_tenant_cannot_be_chosen_by_the_caller(client: httpx.AsyncClient) -> None:
    r = await client.get(
        "/api/v1/search",
        params={"q": "payroll", "tenant_id": "tenant-other"},
        headers={"X-Tenant-ID": "tenant-other"},
    )
    assert r.json()["results"] == []


@pytest.mark.parametrize(
    "params",
    [{}, {"q": "x", "limit": 0}, {"q": "x", "limit": 21}, {"q": "   "}],
)
async def test_invalid_requests_get_422(
    client: httpx.AsyncClient, params: dict[str, object]
) -> None:
    assert (await client.get("/api/v1/search", params=params)).status_code == 422  # type: ignore[arg-type]


async def test_retriever_can_be_injected_at_the_composition_root() -> None:
    async with http(make_app(retriever=FixedRetriever())) as c:
        r = await c.get("/api/v1/search", params={"q": "anything"})
    assert r.json()["results"][0]["chunk_id"] == "fixed"


async def test_dependency_override_swaps_the_service() -> None:
    app = make_app()
    app.dependency_overrides[get_retrieval_service] = lambda: RetrievalService(FixedRetriever())
    async with http(app) as c:
        r = await c.get("/api/v1/search", params={"q": "anything"})
    assert r.json()["results"][0]["chunk_id"] == "fixed"


async def test_production_fails_closed_without_authentication() -> None:
    settings = Settings(
        _env_file=None,
        app_env="production",
        database_url="postgresql+asyncpg://u:p@db.internal:5432/db",
        redis_url="redis://cache.internal:6379/0",
        opensearch_url="https://search.internal:9200",
    )
    async with http(create_app(settings)) as c:
        r = await c.get("/api/v1/search", params={"q": "torque"})
    assert r.status_code == 401
