import re
from collections.abc import Sequence
from dataclasses import dataclass

from ekip_contracts.retrieval import RetrievedChunk


@dataclass(frozen=True)
class StoredChunk:
    tenant_id: str
    chunk_id: str
    document_id: str
    text: str


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class InMemoryRetriever:
    """Temporary adapter: keyword overlap over a list. Replaced by OpenSearch in Phase 5."""

    def __init__(self, chunks: Sequence[StoredChunk]) -> None:
        self._chunks = list(chunks)

    async def search(self, query: str, *, tenant_id: str, limit: int = 5) -> list[RetrievedChunk]:
        terms = _tokens(query)
        if not terms:
            return []
        hits = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                score=len(terms & _tokens(c.text)) / len(terms),
            )
            for c in self._chunks
            if c.tenant_id == tenant_id and terms & _tokens(c.text)  # tenant isolation (I-4)
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:limit]
