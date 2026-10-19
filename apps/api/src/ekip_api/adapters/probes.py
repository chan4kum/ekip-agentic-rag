"""Concrete DependencyProbe adapters. Each opens a short-lived connection per check.

Pools arrive with the first real data access (Phase 2/3), managed in the lifespan hook.
"""

import asyncpg
import httpx
import redis.asyncio as aioredis
from ekip_config.settings import Settings
from ekip_contracts.health import DependencyProbe


class ProbeFailedError(Exception):
    """Dependency answered, but not with a healthy state."""


class PostgresProbe:
    name = "postgres"

    def __init__(self, dsn: str, timeout_s: float = 2.0) -> None:
        self._dsn = dsn
        self._timeout_s = timeout_s

    async def check(self) -> None:
        conn = await asyncpg.connect(self._dsn, timeout=self._timeout_s)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()


class RedisProbe:
    name = "redis"

    def __init__(self, url: str, timeout_s: float = 2.0) -> None:
        self._url = url
        self._timeout_s = timeout_s

    async def check(self) -> None:
        client = aioredis.from_url(
            self._url, socket_connect_timeout=self._timeout_s, socket_timeout=self._timeout_s
        )
        try:
            await client.ping()
        finally:
            await client.aclose()


class OpenSearchProbe:
    name = "opensearch"

    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def check(self) -> None:
        # trust_env=False: internal dependency calls must never go through the egress proxy
        async with httpx.AsyncClient(timeout=self._timeout_s, trust_env=False) as client:
            resp = await client.get(f"{self._base_url}/_cluster/health")
            resp.raise_for_status()
            status = resp.json().get("status")
        if status not in {"green", "yellow"}:
            raise ProbeFailedError(f"cluster status {status!r}")


def build_default_probes(settings: Settings) -> list[DependencyProbe]:
    # asyncpg wants a plain postgresql:// URL, not the SQLAlchemy-style +asyncpg form
    pg_dsn = str(settings.database_url).replace("+asyncpg", "", 1)
    return [
        PostgresProbe(pg_dsn),
        RedisProbe(str(settings.redis_url)),
        OpenSearchProbe(str(settings.opensearch_url)),
    ]
