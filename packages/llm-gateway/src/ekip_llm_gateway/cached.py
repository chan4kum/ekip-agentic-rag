"""Embedding cache: never pay twice for the same (model, task, text).

Keys are hashes, so cached entries reveal nothing about the text and a model or dimension change
automatically misses (the model_version is part of the key).
"""

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Protocol

from ekip_contracts.embedding import Embedder, EmbeddingTask

Vector = tuple[float, ...]


class EmbeddingCache(Protocol):
    async def get_many(self, keys: Sequence[str]) -> dict[str, Vector]: ...

    async def set_many(self, items: Mapping[str, Vector]) -> None: ...


class InMemoryEmbeddingCache:
    """Bounded LRU. A shared (Redis) cache implements the same two methods."""

    def __init__(self, max_entries: int = 100_000) -> None:
        self._max = max_entries
        self._data: OrderedDict[str, Vector] = OrderedDict()

    async def get_many(self, keys: Sequence[str]) -> dict[str, Vector]:
        found: dict[str, Vector] = {}
        for k in keys:
            if k in self._data:
                self._data.move_to_end(k)
                found[k] = self._data[k]
        return found

    async def set_many(self, items: Mapping[str, Vector]) -> None:
        for k, v in items.items():
            self._data[k] = v
            self._data.move_to_end(k)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class CachingEmbedder:
    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def _key(self, text: str, task: EmbeddingTask) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"emb:{self._inner.model_version}:{task.value}:{digest}"

    async def embed(self, texts: Sequence[str], *, task: EmbeddingTask) -> list[Vector]:
        keys = [self._key(t, task) for t in texts]
        hits = await self._cache.get_many(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in hits))
        if missing:
            fresh = await self._inner.embed(missing, task=task)
            computed = {self._key(t, task): v for t, v in zip(missing, fresh, strict=True)}
            await self._cache.set_many(computed)
            hits = {**hits, **computed}
        return [hits[k] for k in keys]
