"""Port for turning text into vectors.

`model_version` identifies the vector SPACE (model + output dimensions, e.g.
`gemini-embedding-001@768`). Vectors from different model versions are never comparable, so the
tag travels with every stored vector (`ChunkEmbedding.model_version`) and changes whenever the
model or its dimensions change.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable


class EmbeddingTask(StrEnum):
    DOCUMENT = "document"  # text that will be indexed
    QUERY = "query"  # a user question searched against the index


class EmbeddingError(Exception):
    """Embedding failed permanently (bad request, bad response, authentication)."""


class EmbeddingUnavailableError(EmbeddingError):
    """Embedding failed transiently and retries were exhausted (rate limit, outage, timeout)."""


class Embedder(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str], *, task: EmbeddingTask) -> list[tuple[float, ...]]:
        """One unit vector per text, in order. Raises only EmbeddingError subclasses."""
        ...


@runtime_checkable
class ScopableEmbedder(Protocol):
    """An embedder whose caches can be partitioned (we use the tenant id).

    Identical text in two tenants must not share a cache entry: a hit would reveal that the other
    tenant holds that text (timing), and a shared store would mix their content.
    """

    def scoped(self, scope: str) -> Embedder: ...
