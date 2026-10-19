import math
from collections.abc import Sequence

from ekip_contracts.embedding import EmbeddingError


def normalize(values: Sequence[float]) -> tuple[float, ...]:
    """L2-normalize. Truncated (Matryoshka) embeddings are not unit length, and cosine/dot-product
    search assumes they are."""
    norm = math.sqrt(sum(v * v for v in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise EmbeddingError("provider returned an unusable vector")
    return tuple(v / norm for v in values)


def validate(values: Sequence[float], dimensions: int) -> None:
    if len(values) != dimensions or not all(math.isfinite(v) for v in values):
        raise EmbeddingError("provider returned a malformed vector")
