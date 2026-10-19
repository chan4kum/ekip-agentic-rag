from ekip_contracts.retrieval import RetrievedChunk, Retriever

MAX_QUERY_CHARS = 1000
MAX_LIMIT = 20


class InvalidQueryError(ValueError):
    """The caller supplied input the service refuses to act on."""


class RetrievalService:
    def __init__(self, retriever: Retriever) -> None:  # constructor injection
        self._retriever = retriever

    async def search(self, query: str, *, tenant_id: str, limit: int = 5) -> list[RetrievedChunk]:
        if not tenant_id:
            raise InvalidQueryError("tenant_id is required")  # fail closed (I-1)
        cleaned = query.strip()
        if not cleaned:
            raise InvalidQueryError("query must not be empty")
        if len(cleaned) > MAX_QUERY_CHARS:
            raise InvalidQueryError("query too long")
        if not 1 <= limit <= MAX_LIMIT:
            raise InvalidQueryError(f"limit must be between 1 and {MAX_LIMIT}")
        return await self._retriever.search(cleaned, tenant_id=tenant_id, limit=limit)
