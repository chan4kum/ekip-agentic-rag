"""Reranker implementations that need no model or network.

These are BASELINES. A cross-encoder (see ekip_llm_gateway.HttpReranker) reads the query and each
passage together and is far better at relevance; the evaluation harness decides whether it earns
its latency and cost on our data.
"""

import re
from collections.abc import Sequence

from ekip_contracts.retrieval import RetrievedChunk


class PassThroughReranker:
    """Keeps the fused order. The 'no reranker' control in evaluations."""

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]:
        return list(candidates[:limit])


_WORD = re.compile(r"[a-z0-9]+")


class TermCoverageReranker:
    """Prefers passages that contain MORE of the distinct query terms (title, section and text).

    Hybrid fusion can rank a passage high because it repeats one rare term; coverage rewards the
    passage that answers the whole question. Ties keep the fused order.
    """

    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], *, limit: int
    ) -> list[RetrievedChunk]:
        terms = set(_WORD.findall(query.lower()))
        if not terms:
            return list(candidates[:limit])

        def coverage(hit: RetrievedChunk) -> float:
            words = set(
                _WORD.findall(f"{hit.title} {' '.join(hit.section_path)} {hit.text}".lower())
            )
            return len(terms & words) / len(terms)

        scored = [(coverage(h), i, h) for i, h in enumerate(candidates)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [h.model_copy(update={"score": c}) for c, _, h in scored[:limit]]
