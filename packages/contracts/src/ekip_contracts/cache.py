"""Cache contracts: what a cached answer looks like and the port that re-checks access.

A cache entry is only ever served after `VisibilityChecker` confirms that the caller can STILL read
every document the entry was built from. The cache key already binds tenant + caller, but the key
is a performance device; the visibility re-check is the security control (invariants I-1, I-4).
"""

from collections.abc import Collection
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ekip_contracts.retrieval import SearchScope


class VisibilityUnavailable(Exception):
    """The checker could not decide (index down, timeout). Caches treat this as a miss."""


class CachedCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    title: str = ""
    source_uri: str = ""
    document_version: int | None = None


class CachedAnswer(BaseModel):
    """A verified, cited answer that may be replayed for an identical request.

    `source_document_ids` lists EVERY document whose text was in the prompt (cited or not): the
    answer is derived from all of them, so the caller must still be able to read all of them.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    citations: tuple[CachedCitation, ...] = ()
    source_document_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _citations_are_sources(self) -> Self:
        missing = {c.document_id for c in self.citations} - set(self.source_document_ids)
        if missing:
            raise ValueError("every cited document must be listed in source_document_ids")
        return self


class VisibilityChecker(Protocol):
    """Port: can `scope` read ALL of `document_ids` right now (tenant, ACL, not deleted)?

    Return False if any is hidden or missing. Raise VisibilityUnavailable (or any exception) if
    it cannot tell; the cache then behaves as a miss and never serves the entry.
    """

    async def all_visible(self, scope: SearchScope, document_ids: Collection[str]) -> bool: ...
