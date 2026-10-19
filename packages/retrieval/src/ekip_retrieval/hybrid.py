"""HybridRetriever: BM25 + vector search, fused with RRF, optionally reranked.

Security: this class never filters access itself. It passes the caller's `SearchScope` to the
searcher, which applies tenant + ACL INSIDE each query. The reranker only ever sees chunks the
searcher already returned, and anything it invents is discarded.

Resilience: the two searches run in parallel, and either may fail without failing the request
(lexical-only or vector-only results, reported in the trace). Both failing is an error. A failing
reranker keeps the fused order. None of these degradations can widen what a caller may see.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ekip_contracts.embedding import Embedder, EmbeddingError, EmbeddingTask, ScopableEmbedder
from ekip_contracts.retrieval import (
    ChunkSearcher,
    Reranker,
    RerankError,
    RetrievedChunk,
    SearchScope,
)
from ekip_contracts.search_index import SearchIndexError
from ekip_observability.logging import get_logger

from ekip_retrieval.fusion import reciprocal_rank_fusion
from ekip_retrieval.query import InvalidQueryError, normalize_query

log = get_logger("ekip.retrieval")
_EXPECTED_FAILURES = (SearchIndexError, EmbeddingError)


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    LEXICAL = "lexical"  # BM25 only (also used as an evaluation control)
    VECTOR = "vector"  # embeddings only (control)


@dataclass(frozen=True)
class HybridConfig:
    mode: SearchMode = SearchMode.HYBRID
    candidates: int = 50  # hits requested from EACH search
    rrf_k: int = 60
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    rerank_top_n: int = 30  # fused candidates handed to the reranker
    max_limit: int = 50  # most results one call may return

    def __post_init__(self) -> None:
        if not (1 <= self.candidates <= 200 and 1 <= self.rerank_top_n <= 200):
            raise ValueError("candidates and rerank_top_n must be between 1 and 200")
        if self.rrf_k < 1 or self.lexical_weight <= 0 or self.vector_weight <= 0:
            raise ValueError("rrf_k >= 1 and weights > 0 required")
        if not 1 <= self.max_limit <= self.candidates:
            raise ValueError("max_limit must be between 1 and candidates")


@dataclass(frozen=True)
class RetrievalTrace:
    """What happened, for tracing and evaluation. Never contains query text or passages."""

    mode: str
    lexical_hits: int = 0
    vector_hits: int = 0
    fused: int = 0
    reranked: bool = False
    degraded: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievedChunk]
    trace: RetrievalTrace


class HybridRetriever:
    def __init__(
        self,
        searcher: ChunkSearcher,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        config: HybridConfig = HybridConfig(),  # noqa: B008 - frozen dataclass
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._searcher = searcher
        self._embedder = embedder
        self._reranker = reranker
        self._config = config
        self._clock = clock

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        return (await self.search_detailed(query, scope=scope, limit=limit)).hits

    async def search_detailed(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> RetrievalResult:
        cfg = self._config
        if not 1 <= limit <= cfg.max_limit:
            raise InvalidQueryError(f"limit must be between 1 and {cfg.max_limit}")
        text = normalize_query(query)
        if not scope.matches_any_acl:  # nobody this caller could be matches an ACL: no work
            return RetrievalResult([], RetrievalTrace(mode=cfg.mode.value))

        t0 = self._clock()
        timings: dict[str, float] = {}
        lexical, vector, degraded = await self._search_both(text, scope, timings)
        if cfg.mode is SearchMode.HYBRID:
            fused = reciprocal_rank_fusion(
                [lexical, vector], k=cfg.rrf_k, weights=[cfg.lexical_weight, cfg.vector_weight]
            )
        else:
            fused = lexical if cfg.mode is SearchMode.LEXICAL else vector
        timings["fuse_ms"] = (self._clock() - t0) * 1000

        reranked = False
        ranked = fused
        if self._reranker is not None and fused:
            t1 = self._clock()
            ranked, reranked, failed = await self._rerank(text, fused, limit)
            if failed:
                degraded.append("rerank_failed")
            timings["rerank_ms"] = (self._clock() - t1) * 1000
        timings["total_ms"] = (self._clock() - t0) * 1000

        if degraded:
            log.warning("retrieval_degraded", degraded=degraded, mode=cfg.mode.value)
        trace = RetrievalTrace(
            mode=cfg.mode.value,
            lexical_hits=len(lexical),
            vector_hits=len(vector),
            fused=len(fused),
            reranked=reranked,
            degraded=tuple(degraded),
            timings_ms=timings,
        )
        return RetrievalResult(ranked[:limit], trace)

    # ---- internals --------------------------------------------------------------------------

    async def _search_both(
        self, text: str, scope: SearchScope, timings: dict[str, float]
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk], list[str]]:
        cfg = self._config
        want_lexical = cfg.mode in (SearchMode.HYBRID, SearchMode.LEXICAL)
        want_vector = cfg.mode in (SearchMode.HYBRID, SearchMode.VECTOR)

        async def lexical() -> list[RetrievedChunk]:
            start = self._clock()
            hits = await self._searcher.lexical_search(text, scope=scope, limit=cfg.candidates)
            timings["lexical_ms"] = (self._clock() - start) * 1000
            return hits

        async def vector() -> list[RetrievedChunk]:
            start = self._clock()
            embedder = (
                self._embedder.scoped(scope.tenant_id)
                if isinstance(self._embedder, ScopableEmbedder)
                else self._embedder
            )
            [embedding] = await embedder.embed([text], task=EmbeddingTask.QUERY)
            timings["embed_ms"] = (self._clock() - start) * 1000
            hits = await self._searcher.vector_search(embedding, scope=scope, limit=cfg.candidates)
            timings["vector_ms"] = (self._clock() - start) * 1000 - timings["embed_ms"]
            return hits

        results = await asyncio.gather(
            lexical() if want_lexical else _nothing(),
            vector() if want_vector else _nothing(),
            return_exceptions=True,
        )
        wanted = (want_lexical, want_vector)
        hits: list[list[RetrievedChunk]] = []
        degraded: list[str] = []
        first_failure: Exception | None = None
        for name, result in zip(("lexical", "vector"), results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, _EXPECTED_FAILURES):
                    raise result  # a bug or cancellation is never "degraded service"
                first_failure = first_failure or result
                degraded.append(f"{name}_failed")
                hits.append([])
            else:
                hits.append(result)
        if first_failure is not None and len(degraded) == sum(wanted):
            raise first_failure  # every search we needed failed
        return hits[0], hits[1], degraded

    async def _rerank(
        self, text: str, fused: list[RetrievedChunk], limit: int
    ) -> tuple[list[RetrievedChunk], bool, bool]:
        assert self._reranker is not None  # noqa: S101 - checked by the caller
        candidates = fused[: self._config.rerank_top_n]
        try:
            out = await self._reranker.rerank(text, candidates, limit=limit)
        except RerankError:
            return fused, False, True
        allowed = {c.chunk_id: c for c in candidates}
        seen: set[str] = set()
        safe: list[RetrievedChunk] = []
        for hit in out:  # a reranker may reorder and rescore, never add or repeat
            if hit.chunk_id in allowed and hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                safe.append(hit)
        return safe, True, False


async def _nothing() -> list[RetrievedChunk]:
    return []
