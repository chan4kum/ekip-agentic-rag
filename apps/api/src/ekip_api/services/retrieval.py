from ekip_contracts.retrieval import RetrievedChunk, Retriever, SearchScope
from ekip_retrieval import MAX_QUERY_CHARS, InvalidQueryError

MAX_LIMIT = 20  # the API's own cap; the retriever enforces its (larger) one as well

__all__ = ["MAX_LIMIT", "MAX_QUERY_CHARS", "InvalidQueryError", "RetrievalService"]


class RetrievalService:
    def __init__(self, retriever: Retriever) -> None:  # constructor injection
        self._retriever = retriever

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        if not 1 <= limit <= MAX_LIMIT:
            raise InvalidQueryError(f"limit must be between 1 and {MAX_LIMIT}")
        # query validation/normalisation and the ACL scope are enforced by the retriever itself
        return await self._retriever.search(query, scope=scope, limit=limit)
