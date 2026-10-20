"""Metric hook for caches: a one-method port the observability layer implements."""

from collections import Counter
from typing import Literal, Protocol

Outcome = Literal["hit", "miss", "stale", "blocked", "corrupt", "error", "skipped"]


class CacheMetrics(Protocol):
    """`cache` names the layer ("response", "retrieval"); `outcome` is what happened.

    hit      served from cache after the visibility re-check passed
    miss     nothing stored (or a stale-version key): computed fresh
    stale    entry found but the caller can no longer read its documents: dropped
    blocked  not cacheable / not verifiable (no ACL identity, visibility check failed): computed
    corrupt  stored bytes were oversized, undecodable or bound to a different caller: ignored
    error    the backing store failed: computed without caching
    skipped  result not stored (policy, size, too many documents)
    """

    def record(self, cache: str, outcome: Outcome) -> None: ...


class NullCacheMetrics:
    def record(self, cache: str, outcome: Outcome) -> None:
        return None


class CounterCacheMetrics:
    """In-process counters (tests, local runs). Production wires a Prometheus implementation."""

    def __init__(self) -> None:
        self.counts: Counter[tuple[str, str]] = Counter()

    def record(self, cache: str, outcome: Outcome) -> None:
        self.counts[(cache, outcome)] += 1

    def get(self, cache: str, outcome: Outcome) -> int:
        return self.counts[(cache, outcome)]
