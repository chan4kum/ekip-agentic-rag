"""The shared engine behind every cache layer.

Order of events for `get_or_compute`:
  1. the caller must have an ACL identity (else: compute, never cache)
  2. read the tenant's corpus generation (store down: compute, never cache)
  3. build a key binding tenant, caller, query, generation and all versions
  4. read; reject oversized / undecodable / differently-bound bytes
  5. RE-CHECK that the caller can still read every source document (fail closed)
  6. on a miss, run the computation once per key (single-flight) and store the result
A cache problem at any step means "compute", never an error to the user.
"""

import contextlib
import json
from collections.abc import Awaitable, Callable, Collection, Sequence
from typing import Any

from ekip_contracts.cache import VisibilityChecker
from ekip_contracts.retrieval import SearchScope
from pydantic import TypeAdapter

from ekip_caching.errors import CacheStoreError
from ekip_caching.generation import CorpusGenerations
from ekip_caching.keys import DEFAULT_PREFIX, build_key, document_tag, scope_fingerprint
from ekip_caching.metrics import CacheMetrics, NullCacheMetrics
from ekip_caching.singleflight import SingleFlight
from ekip_caching.store import CacheStore

ENVELOPE_VERSION = 1
DEFAULT_MAX_BYTES = 256 * 1024
DEFAULT_MAX_TAGS = 256


class GuardedCache[T]:
    def __init__(
        self,
        *,
        layer: str,
        value_type: Any,
        source_documents: Callable[[T], Collection[str]],
        store: CacheStore,
        generations: CorpusGenerations,
        visibility: VisibilityChecker,
        ttl_s: int,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_tags: int = DEFAULT_MAX_TAGS,
        metrics: CacheMetrics | None = None,
        flight: SingleFlight | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        if ttl_s < 1:
            raise ValueError("ttl_s must be >= 1")
        if max_bytes < 1:
            raise ValueError("max_bytes must be >= 1")
        self.layer = layer
        self._adapter: TypeAdapter[T] = TypeAdapter(value_type)
        self._docs = source_documents
        self._store = store
        self._generations = generations
        self._visibility = visibility
        self._ttl = ttl_s
        self._max_bytes = max_bytes
        self._max_tags = max_tags
        self._metrics = metrics or NullCacheMetrics()
        self._flight = flight or SingleFlight()
        self._prefix = prefix

    # -- encoding -------------------------------------------------------------------------

    def _encode(self, value: T, bind: str) -> bytes:
        envelope = {
            "v": ENVELOPE_VERSION,
            "layer": self.layer,
            "bind": bind,
            "data": self._adapter.dump_python(value, mode="json"),
        }
        return json.dumps(envelope, separators=(",", ":")).encode()

    def _decode(self, blob: bytes, bind: str) -> T | None:
        if len(blob) > self._max_bytes:
            return None
        try:
            envelope = json.loads(blob)
            if (
                not isinstance(envelope, dict)
                or envelope.get("v") != ENVELOPE_VERSION
                or envelope.get("layer") != self.layer
                or envelope.get("bind") != bind
            ):
                return None
            return self._adapter.validate_python(envelope["data"])
        except (ValueError, KeyError, TypeError):
            return None

    # -- public ---------------------------------------------------------------------------

    async def get_or_compute(
        self,
        scope: SearchScope,
        parts: Sequence[tuple[str, str]],
        compute: Callable[[], Awaitable[T]],
        *,
        should_cache: Callable[[T], bool] = lambda _value: True,
    ) -> tuple[T, bool]:
        """Return (value, served_from_cache)."""
        if not scope.matches_any_acl:
            self._metrics.record(self.layer, "blocked")
            return await compute(), False
        try:
            generation = await self._generations.current(scope.tenant_id)
        except CacheStoreError:
            self._metrics.record(self.layer, "error")
            return await compute(), False
        fingerprint = scope_fingerprint(scope)
        key = build_key(
            layer=self.layer,
            tenant_id=scope.tenant_id,
            parts=[
                ("scope", fingerprint),
                *parts,
                ("corpus_generation", str(generation)),
            ],
            prefix=self._prefix,
        )
        bind = f"{scope.tenant_id}|{fingerprint}"

        cached = await self._lookup(scope, key, bind)
        if cached is not None:
            return cached, True

        async def leader() -> T:
            value = await compute()
            if should_cache(value):
                await self._save(scope, key, bind, value)
            else:
                self._metrics.record(self.layer, "skipped")
            return value

        return await self._flight.run(key, leader), False

    # -- internals ------------------------------------------------------------------------

    async def _lookup(self, scope: SearchScope, key: str, bind: str) -> T | None:
        try:
            blob = await self._store.get(key)
        except CacheStoreError:
            self._metrics.record(self.layer, "error")
            return None
        if blob is None:
            self._metrics.record(self.layer, "miss")
            return None
        value = self._decode(blob, bind)
        if value is None:
            self._metrics.record(self.layer, "corrupt")
            return None
        documents = set(self._docs(value))
        if documents:
            try:
                visible = await self._visibility.all_visible(scope, documents)
            except Exception:  # noqa: BLE001 - ANY failure to verify access means: do not serve
                self._metrics.record(self.layer, "blocked")
                return None
            if not visible:
                self._metrics.record(self.layer, "stale")
                with contextlib.suppress(CacheStoreError):
                    await self._store.delete(key)
                return None
        self._metrics.record(self.layer, "hit")
        return value

    async def _save(self, scope: SearchScope, key: str, bind: str, value: T) -> None:
        documents = sorted(set(self._docs(value)))
        if len(documents) > self._max_tags:
            self._metrics.record(self.layer, "skipped")  # could not be purged per document
            return
        blob = self._encode(value, bind)
        if len(blob) > self._max_bytes:
            self._metrics.record(self.layer, "skipped")
            return
        tags = [document_tag(scope.tenant_id, d, prefix=self._prefix) for d in documents]
        try:
            await self._store.set(key, blob, ttl_s=self._ttl, tags=tags)
        except CacheStoreError:
            self._metrics.record(self.layer, "error")
