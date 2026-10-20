"""Test doubles and builders shared by the cache tests."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from ekip_caching import (
    CacheInvalidator,
    CacheStoreError,
    CacheVersions,
    CorpusGenerations,
    CounterCacheMetrics,
    MemoryCacheStore,
    ResponseCache,
    RetrievalCache,
)
from ekip_contracts.cache import CachedAnswer, CachedCitation, VisibilityUnavailable
from ekip_contracts.retrieval import RetrievedChunk, SearchScope

VERSIONS = CacheVersions(
    retrieval_config="rc1", prompt="answer-v2", model="gpt-x", output_policy="op1"
)


def scope(tenant: str = "acme", user: str = "alice", *groups: str) -> SearchScope:
    return SearchScope(tenant_id=tenant, user=user, groups=frozenset(groups))


def answer(*docs: str, text: str = "The torque is 40 Nm [1].") -> CachedAnswer:
    return CachedAnswer(
        answer=text,
        citations=tuple(CachedCitation(chunk_id=f"{d}:c0", document_id=d) for d in docs[:1]),
        source_document_ids=tuple(docs),
    )


def hit(doc: str, chunk: str = "c0") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=f"{doc}:{chunk}", document_id=doc, text="text", score=1.0)


@dataclass
class FakeVisibility:
    """Documents in `hidden` are invisible to everyone; `broken` simulates an outage."""

    hidden: set[str] = field(default_factory=set)
    broken: bool = False
    calls: list[tuple[SearchScope, frozenset[str]]] = field(default_factory=list)

    async def all_visible(self, scope: SearchScope, document_ids: Collection[str]) -> bool:
        self.calls.append((scope, frozenset(document_ids)))
        if self.broken:
            raise VisibilityUnavailable("index down")
        return not (self.hidden & set(document_ids))


class FailingStore:
    """A store whose backend is down: every operation raises CacheStoreError."""

    async def get(self, key: str) -> bytes | None:
        raise CacheStoreError("down")

    async def set(self, key: str, value: bytes, *, ttl_s: int, tags: Sequence[str] = ()) -> None:
        raise CacheStoreError("down")

    async def delete(self, key: str) -> None:
        raise CacheStoreError("down")

    async def delete_tag(self, tag: str) -> int:
        raise CacheStoreError("down")

    async def incr(self, key: str) -> int:
        raise CacheStoreError("down")

    async def get_or_init_counter(self, key: str, initial: int) -> int:
        raise CacheStoreError("down")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Stack:
    store: MemoryCacheStore
    generations: CorpusGenerations
    visibility: FakeVisibility
    metrics: CounterCacheMetrics
    clock: Clock
    response: ResponseCache
    retrieval: RetrievalCache
    invalidator: CacheInvalidator


def make_stack(
    *, versions: CacheVersions = VERSIONS, ttl_s: int = 60, max_bytes: int = 64 * 1024
) -> Stack:
    clock = Clock()
    store = MemoryCacheStore(clock=clock)
    generations = CorpusGenerations(store)
    visibility = FakeVisibility()
    metrics = CounterCacheMetrics()
    return Stack(
        store=store,
        generations=generations,
        visibility=visibility,
        metrics=metrics,
        clock=clock,
        response=ResponseCache(
            store=store,
            generations=generations,
            visibility=visibility,
            versions=versions,
            ttl_s=ttl_s,
            max_bytes=max_bytes,
            metrics=metrics,
        ),
        retrieval=RetrievalCache(
            store=store,
            generations=generations,
            visibility=visibility,
            config_version=versions.retrieval_config,
            ttl_s=ttl_s,
            metrics=metrics,
        ),
        invalidator=CacheInvalidator(store, generations),
    )


class Counter:
    """An awaitable computation that counts how often it really ran."""

    def __init__(self, result: CachedAnswer) -> None:
        self.result = result
        self.calls = 0

    async def __call__(self) -> CachedAnswer:
        self.calls += 1
        return self.result
