"""Offline micro-benchmark of the cache hot paths (no network, no Redis, no model).

    uv run --no-sync python scripts/bench_cache.py [iterations]

It times key building and the in-memory HIT path (key, store read, decode, visibility call).
The numbers describe THIS machine and an in-process fake; they say nothing about production
latency, where Redis round trips and the real visibility query dominate. Use them only to catch
regressions in our own code.
"""

import asyncio
import statistics
import sys
import time
from collections.abc import Awaitable, Callable, Collection

from ekip_caching import (
    CacheVersions,
    CorpusGenerations,
    MemoryCacheStore,
    ResponseCache,
    build_key,
    normalise_query,
    scope_fingerprint,
)
from ekip_contracts.cache import CachedAnswer, CachedCitation
from ekip_contracts.retrieval import SearchScope


class AllowAll:
    async def all_visible(self, scope: SearchScope, document_ids: Collection[str]) -> bool:
        return True


def timed(fn: Callable[[], object], n: int) -> list[float]:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


async def timed_async(fn: Callable[[], Awaitable[object]], n: int) -> list[float]:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        await fn()
        samples.append(time.perf_counter() - t0)
    return samples


def report(name: str, samples: list[float]) -> None:
    us = sorted(s * 1e6 for s in samples)
    p95 = us[int(len(us) * 0.95) - 1]
    print(f"{name:<34} median {statistics.median(us):8.1f} us   p95 {p95:8.1f} us   n={len(us)}")


async def main(n: int) -> None:
    scope = SearchScope(tenant_id="acme", user="alice", groups=frozenset({"hr", "eng", "all"}))
    query = "What changed between revision 6 and revision 7 of the hydraulic maintenance manual?"

    report("normalise_query", timed(lambda: normalise_query(query), n))
    report("scope_fingerprint", timed(lambda: scope_fingerprint(scope), n))
    report(
        "build_key",
        timed(
            lambda: build_key(
                layer="response",
                tenant_id="acme",
                parts=[("query", normalise_query(query)), ("corpus_generation", "42")],
            ),
            n,
        ),
    )

    store = MemoryCacheStore()
    cache = ResponseCache(
        store=store,
        generations=CorpusGenerations(store),
        visibility=AllowAll(),
        versions=CacheVersions(
            retrieval_config="rc1", prompt="answer-v2", model="m1", output_policy="op1"
        ),
    )
    docs = tuple(f"doc-{i}" for i in range(6))
    value = CachedAnswer(
        answer="Section 3.2 was revised [1]. " * 40,
        citations=tuple(CachedCitation(chunk_id=f"{d}:c0", document_id=d) for d in docs[:3]),
        source_document_ids=docs,
    )

    async def compute() -> CachedAnswer:
        return value

    async def call() -> object:
        return await cache.get_or_compute(scope, query, compute, authz_version="a1")

    await call()  # populate
    report("response cache hit (in-memory)", await timed_async(call, n))


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20_000))
