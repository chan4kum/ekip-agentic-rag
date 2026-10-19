import pytest
from ekip_api.services.retrieval import MAX_LIMIT, InvalidQueryError, RetrievalService
from ekip_contracts.retrieval import RetrievedChunk, SearchScope

SCOPE = SearchScope("t1", user="alice", groups=frozenset({"eng"}))


class SpyRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SearchScope, int]] = []

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.calls.append((query, scope, limit))
        return [RetrievedChunk(chunk_id="x", document_id="d", text="t", score=1.0)]


async def test_passes_the_callers_scope_and_limit_through_unchanged() -> None:
    spy = SpyRetriever()
    results = await RetrievalService(spy).search("hello", scope=SCOPE, limit=3)
    assert spy.calls == [("hello", SCOPE, 3)]
    assert len(results) == 1


@pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1])
async def test_out_of_range_limits_are_rejected_before_the_retriever(limit: int) -> None:
    spy = SpyRetriever()
    with pytest.raises(InvalidQueryError):
        await RetrievalService(spy).search("ok", scope=SCOPE, limit=limit)
    assert spy.calls == []
