"""Response and retrieval caches with safe, versioned keys.

Safety model: a cache entry is bound to tenant + caller + corpus generation + every version that
shapes the answer, and is re-checked against the caller's current access on every hit. The cache
is an optimisation only: any failure falls back to computing the answer.
"""

from ekip_caching.errors import CacheInvalidationError, CacheStoreError
from ekip_caching.generation import CorpusGenerations
from ekip_caching.invalidation import CacheInvalidator
from ekip_caching.keys import CacheVersions, build_key, normalise_query, scope_fingerprint
from ekip_caching.metrics import CacheMetrics, CounterCacheMetrics, NullCacheMetrics, Outcome
from ekip_caching.redis_store import RedisCacheStore
from ekip_caching.response_cache import ResponseCache
from ekip_caching.retrieval_cache import CachedRetriever, RetrievalCache
from ekip_caching.singleflight import SingleFlight
from ekip_caching.store import CacheStore, MemoryCacheStore

__all__ = [
    "CacheInvalidationError",
    "CacheInvalidator",
    "CacheMetrics",
    "CacheStore",
    "CacheStoreError",
    "CacheVersions",
    "CachedRetriever",
    "CorpusGenerations",
    "CounterCacheMetrics",
    "MemoryCacheStore",
    "NullCacheMetrics",
    "Outcome",
    "RedisCacheStore",
    "ResponseCache",
    "RetrievalCache",
    "SingleFlight",
    "build_key",
    "normalise_query",
    "scope_fingerprint",
]
