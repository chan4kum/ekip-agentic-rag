"""Cache errors. A cache failure never reaches the user: layers catch these and recompute."""


class CacheStoreError(Exception):
    """The backing store (Redis) could not complete an operation."""


class CacheInvalidationError(Exception):
    """An invalidation could not be completed; the caller must retry (stale data may be served)."""
