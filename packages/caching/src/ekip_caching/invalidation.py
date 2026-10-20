"""Invalidation entry points for ingestion and authorization.

delete / permission change   bump the tenant's generation FIRST (every cached answer of the
                             tenant becomes unreachable immediately), then purge the entries
                             tagged with the document to reclaim memory
new version published        purge entries tagged with the document (so no stale citation is
                             served); bump too only if `bump_on_publish` is set. Without a bump,
                             queries that merely WOULD rank a new document still see old results
                             until TTL; that is a freshness trade-off, not a security one.
Failures raise CacheInvalidationError: the caller (workflow) must retry.
"""

from ekip_caching.errors import CacheInvalidationError, CacheStoreError
from ekip_caching.generation import CorpusGenerations
from ekip_caching.keys import DEFAULT_PREFIX, document_tag
from ekip_caching.store import CacheStore


class CacheInvalidator:
    def __init__(
        self,
        store: CacheStore,
        generations: CorpusGenerations,
        *,
        bump_on_publish: bool = False,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        self._store = store
        self._generations = generations
        self._bump_on_publish = bump_on_publish
        self._prefix = prefix
        self.purge_errors = 0  # purge failures after a successful bump (memory only, not safety)

    async def document_deleted(self, tenant_id: str, document_id: str) -> None:
        await self._revoke(tenant_id, document_id)

    async def acl_changed(self, tenant_id: str, document_id: str) -> None:
        await self._revoke(tenant_id, document_id)

    async def document_published(self, tenant_id: str, document_id: str) -> None:
        try:
            if self._bump_on_publish:
                await self._generations.bump(tenant_id)
            await self._store.delete_tag(document_tag(tenant_id, document_id, prefix=self._prefix))
        except CacheStoreError as exc:
            raise CacheInvalidationError("publish invalidation failed") from exc

    async def tenant_reset(self, tenant_id: str) -> None:
        """Retire every cached entry of a tenant (policy change, re-index, incident)."""
        try:
            await self._generations.bump(tenant_id)
        except CacheStoreError as exc:
            raise CacheInvalidationError("tenant reset failed") from exc

    async def _revoke(self, tenant_id: str, document_id: str) -> None:
        try:
            await self._generations.bump(tenant_id)
        except CacheStoreError as exc:
            raise CacheInvalidationError("generation bump failed") from exc
        try:  # entries are already unreachable; this only frees memory
            await self._store.delete_tag(document_tag(tenant_id, document_id, prefix=self._prefix))
        except CacheStoreError:
            self.purge_errors += 1
