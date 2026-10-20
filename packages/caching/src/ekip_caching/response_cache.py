"""Exact-match cache of verified, cited answers (cache layer 4 of the architecture)."""

from collections.abc import Awaitable, Callable

from ekip_contracts.cache import CachedAnswer, VisibilityChecker
from ekip_contracts.retrieval import SearchScope

from ekip_caching.generation import CorpusGenerations
from ekip_caching.guarded import DEFAULT_MAX_BYTES, DEFAULT_MAX_TAGS, GuardedCache
from ekip_caching.keys import DEFAULT_PREFIX, CacheVersions, normalise_query
from ekip_caching.metrics import CacheMetrics
from ekip_caching.singleflight import SingleFlight
from ekip_caching.store import CacheStore


def _grounded(answer: CachedAnswer) -> bool:
    """Only answers built from sources are replayed; refusals/"no evidence" are recomputed so a
    document ingested a minute later is never masked."""
    return bool(answer.source_document_ids)


class ResponseCache:
    def __init__(
        self,
        *,
        store: CacheStore,
        generations: CorpusGenerations,
        visibility: VisibilityChecker,
        versions: CacheVersions,
        ttl_s: int = 3600,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_tags: int = DEFAULT_MAX_TAGS,
        metrics: CacheMetrics | None = None,
        flight: SingleFlight | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        self._versions = versions
        self._core: GuardedCache[CachedAnswer] = GuardedCache(
            layer="response",
            value_type=CachedAnswer,
            source_documents=lambda a: a.source_document_ids,
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
        compute: Callable[[], Awaitable[CachedAnswer]],
        *,
        authz_version: str,
        should_cache: Callable[[CachedAnswer], bool] = _grounded,
    ) -> tuple[CachedAnswer, bool]:
        """Return (answer, served_from_cache). `authz_version` is the version of the policy /
        ACL model in force (bump it on a policy change to retire every entry)."""
        parts = [
            ("authorization_scope_version", authz_version),
            ("query", normalise_query(query)),
            *self._versions.as_parts(),
        ]
        return await self._core.get_or_compute(scope, parts, compute, should_cache=should_cache)
