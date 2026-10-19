import asyncio
from typing import Any

import pytest
from ekip_identity.jwks import CachedJwks, KeyNotFoundError, KeysUnavailableError
from idp_testkit import jwks, make_key, public_jwk

K1 = public_jwk(make_key(), "k1")
K2 = public_jwk(make_key(), "k2")
ENC = public_jwk(make_key(), "enc1", use="enc", alg="RSA-OAEP")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeIdp:
    def __init__(self, *docs: Any) -> None:
        self.docs = list(docs)
        self.calls = 0
        self.fail = False
        self.delay = 0.0

    async def __call__(self) -> Any:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("idp down")
        return self.docs[min(self.calls - 1, len(self.docs) - 1)]


def _cache(idp: FakeIdp, clock: Clock, **kw: Any) -> CachedJwks:
    return CachedJwks(idp, clock=clock, **kw)


async def test_fetches_once_then_serves_from_cache() -> None:
    idp, clock = FakeIdp(jwks(K1)), Clock()
    cache = _cache(idp, clock)
    await cache.key_for("k1")
    await cache.key_for("k1")
    assert idp.calls == 1


async def test_encryption_keys_are_dropped() -> None:
    cache = _cache(FakeIdp(jwks(K1, ENC)), Clock())
    await cache.key_for("k1")
    with pytest.raises(KeyNotFoundError):
        await cache.key_for("enc1")


async def test_refetches_after_ttl() -> None:
    idp, clock = FakeIdp(jwks(K1)), Clock()
    cache = _cache(idp, clock, ttl_s=100)
    await cache.key_for("k1")
    clock.now += 101
    await cache.key_for("k1")
    assert idp.calls == 2


async def test_rotation_new_kid_triggers_refetch() -> None:
    idp, clock = FakeIdp(jwks(K1), jwks(K1, K2)), Clock()
    cache = _cache(idp, clock, min_refetch_s=30)
    await cache.key_for("k1")
    clock.now += 31
    assert (await cache.key_for("k2")) is not None
    assert idp.calls == 2


async def test_unknown_kid_flood_is_rate_limited() -> None:
    idp, clock = FakeIdp(jwks(K1)), Clock()
    cache = _cache(idp, clock, min_refetch_s=30)
    await cache.key_for("k1")
    for i in range(50):
        with pytest.raises(KeyNotFoundError):
            await cache.key_for(f"random-{i}")
    assert idp.calls <= 2  # initial fetch + at most one permitted refetch


async def test_idp_down_at_startup_fails_closed() -> None:
    idp = FakeIdp(jwks(K1))
    idp.fail = True
    cache = _cache(idp, Clock())
    with pytest.raises(KeysUnavailableError):
        await cache.key_for("k1")


async def test_idp_down_second_call_still_fails_closed_without_hammering() -> None:
    idp = FakeIdp(jwks(K1))
    idp.fail = True
    clock = Clock()
    cache = _cache(idp, clock, min_refetch_s=30)
    for _ in range(5):
        with pytest.raises(KeysUnavailableError):
            await cache.key_for("k1")
    assert idp.calls == 1


async def test_stale_keys_served_when_idp_goes_down() -> None:
    idp, clock = FakeIdp(jwks(K1)), Clock()
    cache = _cache(idp, clock, ttl_s=100, stale_ttl_s=1000)
    await cache.key_for("k1")
    idp.fail = True
    clock.now += 200  # past ttl, within stale window
    assert (await cache.key_for("k1")) is not None


async def test_stale_keys_expire() -> None:
    idp, clock = FakeIdp(jwks(K1)), Clock()
    cache = _cache(idp, clock, ttl_s=100, stale_ttl_s=1000, min_refetch_s=1)
    await cache.key_for("k1")
    idp.fail = True
    clock.now += 2000
    with pytest.raises(KeysUnavailableError):
        await cache.key_for("k1")


async def test_empty_key_set_is_not_trusted() -> None:
    cache = _cache(FakeIdp({"keys": []}), Clock())
    with pytest.raises(KeysUnavailableError):
        await cache.key_for("k1")


async def test_concurrent_misses_share_one_fetch() -> None:
    idp = FakeIdp(jwks(K1))
    idp.delay = 0.05
    cache = _cache(idp, Clock())
    results = await asyncio.gather(*(cache.key_for("k1") for _ in range(20)))
    assert len(results) == 20
    assert idp.calls == 1
