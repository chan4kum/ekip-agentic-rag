import pytest
from ekip_api.services.retrieval import MAX_LIMIT, InvalidQueryError, RetrievalService
from ekip_contracts.retrieval import RetrievedChunk


class SpyRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    async def search(self, query: str, *, tenant_id: str, limit: int = 5) -> list[RetrievedChunk]:
        self.calls.append((query, tenant_id, limit))
        return [RetrievedChunk(chunk_id="x", document_id="d", text="t", score=1.0)]


async def test_trims_query_and_passes_tenant_and_limit_through() -> None:
    spy = SpyRetriever()
    results = await RetrievalService(spy).search("  hello  ", tenant_id="t1", limit=3)
    assert spy.calls == [("hello", "t1", 3)]
    assert len(results) == 1


@pytest.mark.parametrize(
    ("query", "tenant", "limit"),
    [
        ("", "t1", 5),
        ("   ", "t1", 5),
        ("x" * 1001, "t1", 5),
        ("ok", "t1", 0),
        ("ok", "t1", MAX_LIMIT + 1),
        ("ok", "", 5),
    ],
)
async def test_invalid_input_is_rejected_before_the_retriever(
    query: str, tenant: str, limit: int
) -> None:
    spy = SpyRetriever()
    with pytest.raises(InvalidQueryError):
        await RetrievalService(spy).search(query, tenant_id=tenant, limit=limit)
    assert spy.calls == []
