"""Deterministic offline embedder (hashed bag of words). For tests and local development.

Texts that share words get a higher cosine similarity, so retrieval logic can be tested without a
network or a key. It is NOT a semantic model and must never be used for real data.
"""

import hashlib
import re
from collections.abc import Sequence

from ekip_contracts.embedding import EmbeddingError, EmbeddingTask

from ekip_llm_gateway.vectors import normalize

_WORD = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self._dims = dimensions
        self.calls = 0  # how many embed() calls reached this embedder (handy in cache tests)
        self.texts_embedded = 0

    @property
    def model_version(self) -> str:
        return f"hashing-bow@{self._dims}"

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed(self, texts: Sequence[str], *, task: EmbeddingTask) -> list[tuple[float, ...]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self._dims
        words = _WORD.findall(text.lower())
        if not words:
            raise EmbeddingError("cannot embed text without words")
        for w in words:
            h = hashlib.blake2b(w.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self._dims
            vec[idx] += 1.0 if h[4] & 1 else -1.0
        if not any(vec):  # perfectly cancelling signs: nudge deterministically
            vec[0] = 1.0
        return normalize(vec)
