"""Merge the search results of several sub-questions into one candidate list.

Round-robin by rank, so no single sub-question crowds the others out of the context budget (the
first hit of every task comes before the second hit of any). A chunk found by several tasks is kept
once, at its earliest position. The ContextAssembler then numbers the merged list, so the final
answer cites sources from ONE list that exists in one place.
"""

from collections.abc import Sequence

from ekip_contracts.retrieval import RetrievedChunk


def merge_hits(per_task: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    depth = max((len(hits) for hits in per_task), default=0)
    for rank in range(depth):
        for hits in per_task:
            if rank < len(hits) and hits[rank].chunk_id not in seen:
                seen.add(hits[rank].chunk_id)
                merged.append(hits[rank])
    return merged
