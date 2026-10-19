"""JWKS (JSON Web Key Set) cache: where the validator gets public signing keys."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

import httpx
import jwt
from ekip_observability.logging import get_logger

log = get_logger("ekip.identity.jwks")

Fetcher = Callable[[], Awaitable[Mapping[str, Any]]]
_ALLOWED_KTY = {"RSA"}
_ALLOWED_ALG = {None, "RS256"}


class KeyNotFoundError(Exception):
    """No signing key with that kid (after one permitted refresh)."""


class KeysUnavailableError(Exception):
    """We could not obtain any trustworthy key set (network down, bad document...)."""


class KeyProvider(Protocol):
    async def key_for(self, kid: str) -> jwt.PyJWK: ...


def httpx_fetcher(url: str, timeout_s: float = 3.0) -> Fetcher:
    """Fetch the JWKS over HTTP. The URL comes from configuration, never from a token."""

    async def fetch() -> Mapping[str, Any]:
        # trust_env=False: internal call, must not go through the egress proxy
        async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data: Mapping[str, Any] = resp.json()
            return data

    return fetch


def _parse(document: Mapping[str, Any]) -> dict[str, jwt.PyJWK]:
    """Keep only RSA signature keys with a kid (drops e.g. Keycloak's RSA-OAEP encryption key)."""
    keys: dict[str, jwt.PyJWK] = {}
    for jwk in document.get("keys", []):
        kid = jwk.get("kid")
        if not isinstance(kid, str) or jwk.get("kty") not in _ALLOWED_KTY:
            continue
        if jwk.get("use") not in (None, "sig") or jwk.get("alg") not in _ALLOWED_ALG:
            continue
        try:
            keys[kid] = jwt.PyJWK.from_dict(dict(jwk))
        except jwt.PyJWTError:
            continue
    return keys


class CachedJwks:
    """Caches keys; refreshes on expiry or an unknown kid, but never hammers the IdP.

    - ttl_s: how long a fetched key set is trusted without re-fetching
    - min_refetch_s: minimum gap between fetch ATTEMPTS (an attacker sending random kids must
      not turn us into a request amplifier against the identity provider)
    - stale_ttl_s: if the IdP is down, keep serving a key set no older than this
    """

    def __init__(
        self,
        fetch: Fetcher,
        *,
        ttl_s: float = 600,
        min_refetch_s: float = 30,
        stale_ttl_s: float = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetch = fetch
        self._ttl_s = ttl_s
        self._min_refetch_s = min_refetch_s
        self._stale_ttl_s = stale_ttl_s
        self._clock = clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at: float | None = None
        self._last_attempt: float | None = None
        self._lock = asyncio.Lock()

    def _fresh(self, now: float) -> bool:
        return self._fetched_at is not None and now - self._fetched_at < self._ttl_s

    async def key_for(self, kid: str) -> jwt.PyJWK:
        now = self._clock()
        if self._fresh(now) and kid in self._keys:
            return self._keys[kid]

        recently_tried = (
            self._last_attempt is not None and now - self._last_attempt < self._min_refetch_s
        )
        if self._fresh(now) and recently_tried:
            raise KeyNotFoundError(kid)  # fresh cache, unknown kid, refetch not yet allowed

        async with self._lock:  # one fetch at a time; waiters reuse its result
            now = self._clock()
            if self._fresh(now) and kid in self._keys:
                return self._keys[kid]
            if self._last_attempt is not None and now - self._last_attempt < self._min_refetch_s:
                if kid in self._keys and self._stale_ok(now):
                    return self._keys[kid]
                if not self._keys:
                    raise KeysUnavailableError  # we have never had keys and just failed to get them
                raise KeyNotFoundError(kid)
            self._last_attempt = now
            await self._refresh(now)

        if kid in self._keys:
            return self._keys[kid]
        raise KeyNotFoundError(kid)

    def _stale_ok(self, now: float) -> bool:
        return self._fetched_at is not None and now - self._fetched_at < self._stale_ttl_s

    async def _refresh(self, now: float) -> None:
        try:
            keys = _parse(await self._fetch())
        except Exception as exc:
            log.warning("jwks_fetch_failed", error_type=type(exc).__name__)
            if self._keys and self._stale_ok(now):
                return  # serve the stale set for a while rather than lock everyone out
            raise KeysUnavailableError from exc
        if not keys:
            log.warning("jwks_empty")
            if self._keys and self._stale_ok(now):
                return
            raise KeysUnavailableError
        self._keys = keys
        self._fetched_at = now
