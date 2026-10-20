"""Cache of retrieval results (cache layer 3), plus a Retriever decorator that uses it."""

from collections.abc import Awaitable, Callable

from ekip_contracts.cache import VisibilityChecker
from ekip_contracts.retrieval import RetrievedChunk, Retriever, SearchScope

from ekip_caching.generation import CorpusGenerations
from ekip_caching.guarded import DEFAULT_MAX_BYTES, DEFAULT_MAX_TAGS, GuardedCache
from ekip_caching.keys import DEFAULT_PREFIX, normalise_query
from ekip_caching.metrics import CacheMetrics
from ekip_caching.singleflight import SingleFlight
from ekip_caching.store import CacheStore


class RetrievalCache:
    def __init__(
        self,
        *,
        store: CacheStore,
        generations: CorpusGenerations,
        visibility: VisibilityChecker,
        config_version: str,
        ttl_s: int = 300,
        max_bytes: int = DEFAULT_MAX_BYTES * 4,
        max_tags: int = DEFAULT_MAX_TAGS,
        metrics: CacheMetrics | None = None,
        flight: SingleFlight | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        if not config_version.strip():
            raise ValueError("config_version must be non-empty")
        self._config_version = config_version
        self._core: GuardedCache[list[RetrievedChunk]] = GuardedCache(
            layer="retrieval",
            value_type=list[RetrievedChunk],
            source_documents=lambda hits: [h.document_id for h in hits],
            store=store,
            generations=generations,
            visibility=visibility,
            ttl_s=ttl_s,
            max_bytes=max_bytes,
            max_tags=max_tags,
            metrics=metrics,
            flight=flight,
            prefix=prefix,
        )

    async def get_or_compute(
        self,
        scope: SearchScope,
        query: str,
        limit: int,
        compute: Callable[[], Awaitable[list[RetrievedChunk]]],
    ) -> tuple[list[RetrievedChunk], bool]:
        parts = [
            ("query", normalise_query(query)),
            ("limit", str(limit)),
            ("retrieval_config_version", self._config_version),
        ]
        # Empty results are not stored: "nothing found" must not mask a document ingested later.
        return await self._core.get_or_compute(
            scope, parts, compute, should_cache=lambda hits: bool(hits)
        )


class CachedRetriever:
    """Drop-in `Retriever` that consults the cache first. Wrap the real retriever with it."""

    def __init__(self, inner: Retriever, cache: RetrievalCache) -> None:
        self._inner = inner
        self._cache = cache

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        hits, _ = await self._cache.get_or_compute(
            scope, query, limit, lambda: self._inner.search(query, scope=scope, limit=limit)
        )
        return hits
