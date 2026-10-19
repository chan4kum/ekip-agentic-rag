"""Reciprocal Rank Fusion (Cormack et al., 2009).

Each ranking contributes weight / (k + rank) for every chunk it contains. Using RANKS rather than
raw scores matters: BM25 scores (unbounded) and cosine similarities (0..1) are not comparable, and
RRF needs no normalisation. k=60 is the standard damping constant: it stops a single first place
in one list from outweighing consistent top-10 placements in both.
"""

from collections.abc import Sequence

from ekip_contracts.retrieval import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Merge best-first rankings into one. The result's `score` is the fused RRF score.

    A chunk's data is taken from the first ranking that contains it. Duplicates inside one ranking
    count once (their best rank). Ties break on best rank, then chunk id, so output is stable.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if weights is not None and (len(weights) != len(rankings) or any(w <= 0 for w in weights)):
        raise ValueError("weights must be positive, one per ranking")
    w = list(weights) if weights is not None else [1.0] * len(rankings)

    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for ranking, weight in zip(rankings, w, strict=True):
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + weight / (k + rank)
            best_rank[hit.chunk_id] = min(best_rank.get(hit.chunk_id, rank), rank)
            first_seen.setdefault(hit.chunk_id, hit)
    order = sorted(fused, key=lambda cid: (-fused[cid], best_rank[cid], cid))
    return [first_seen[cid].model_copy(update={"score": fused[cid]}) for cid in order]
