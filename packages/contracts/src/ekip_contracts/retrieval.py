"""Retrieval contracts: who is asking, what comes back, and the ports around the search engine.

`SearchScope` is the only way retrieval learns who is asking. It is built from a verified
`SecurityContext` (never from request parameters), and every search adapter must apply it INSIDE
the query, so documents the caller may not see are never even candidates (invariants I-1, I-4, I-6).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ekip_contracts.documents import PRINCIPAL, TENANT_ID, ContractError
from ekip_contracts.identity import SecurityContext


@dataclass(frozen=True)
class SearchScope:
    """Tenant + the identities a document ACL can name. A caller with neither a user nor a group
    matches no ACL entry, so they see nothing (an empty ACL also matches nobody)."""

    tenant_id: str
    user: str = ""
    groups: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not TENANT_ID.fullmatch(self.tenant_id):
            raise ContractError("invalid tenant_id")
        if self.user and not PRINCIPAL.fullmatch(self.user):
            raise ContractError("invalid user principal")
        if not all(PRINCIPAL.fullmatch(g) for g in self.groups):
            raise ContractError("invalid group name")

    @classmethod
    def from_context(cls, context: SecurityContext) -> "SearchScope":
        return cls(tenant_id=context.tenant_id, user=context.principal or "", groups=context.groups)

    @property
    def matches_any_acl(self) -> bool:
        return bool(self.user or self.groups)


class RetrievedChunk(BaseModel):
    """One search hit. `score` is only comparable within the list it came from."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    text: str
    score: float = Field(ge=0)
    document_version: int | None = None
    title: str = ""
    source_uri: str = ""
    section_path: tuple[str, ...] = ()
    page_number: int | None = None
    content_hash: str | None = None
    kind: str = "text"


class ChunkSearcher(Protocol):
    """Read side of the chunk index. Both methods apply tenant + ACL filtering inside the query
    and return at most `limit` hits, best first. Raise SearchIndexError on failure."""

    async def lexical_search(
        self, query: str, *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        """BM25 over title, section path and content."""
        ...

    async def vector_search(
        self, vector: Sequence[float], *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        """Nearest neighbours of `vector` (must be from the index's embedding model)."""
        ...


class RerankError(Exception):
    """The reranker could not score the candidates (outage, bad response). Callers keep the
    fused order instead of failing the search."""


class Reranker(Protocol):
    """Re-orders candidates by relevance to the query. Never adds candidates, never drops access
    checks (it only sees what the searcher already returned)."""

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]: ...


class Retriever(Protocol):
    """Port: anything that can search chunks on behalf of ONE caller."""

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]: ...
