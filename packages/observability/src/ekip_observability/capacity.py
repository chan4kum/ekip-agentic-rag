"""Capacity arithmetic from masterplan section 6, as code so the numbers can be re-run.

These are RAW sizing inputs only. They deliberately exclude ANN structures, BM25 postings,
document text, replicas, segment overhead and re-index headroom - the real footprint must be
measured on a representative corpus (see docs/capacity-model.md).
"""

from dataclasses import dataclass

BYTES_PER_FLOAT32 = 4
GB = 10**9  # the masterplan uses decimal gigabytes (81.92 GB)


@dataclass(frozen=True)
class CorpusAssumptions:
    documents: int = 1_000_000
    chunks_per_document: int = 20
    embedding_dimensions: int = 1024
    bytes_per_dimension: int = BYTES_PER_FLOAT32
    daily_changed_documents: int = 10_000

    def __post_init__(self) -> None:
        for name in (
            "documents",
            "chunks_per_document",
            "embedding_dimensions",
            "bytes_per_dimension",
            "daily_changed_documents",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")

    @property
    def total_chunks(self) -> int:
        return self.documents * self.chunks_per_document

    @property
    def raw_vector_bytes(self) -> int:
        return self.total_chunks * self.embedding_dimensions * self.bytes_per_dimension

    @property
    def raw_vector_gb(self) -> float:
        return self.raw_vector_bytes / GB

    @property
    def max_changed_chunks_per_day(self) -> int:
        """Upper bound: every chunk of every changed document is re-embedded."""
        return self.daily_changed_documents * self.chunks_per_document


def replicated_vector_gb(assumptions: CorpusAssumptions, *, replicas: int) -> float:
    """Raw vector bytes held across `replicas` extra copies (primary + replicas)."""
    if replicas < 0:
        raise ValueError("replicas must be >= 0")
    return assumptions.raw_vector_gb * (1 + replicas)


def backfill_hours(total_chunks: int, embeddings_per_second: float) -> float:
    """Time to embed a corpus at a MEASURED throughput (never a guessed one)."""
    if embeddings_per_second <= 0:
        raise ValueError("embeddings_per_second must be > 0")
    return total_chunks / embeddings_per_second / 3600
