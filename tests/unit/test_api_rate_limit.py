import httpx
from ekip_api.main import create_app
from ekip_config.settings import Settings
from ekip_contracts.ratelimit import (
    RateLimitDecision,
    RateLimits,
    RateLimitUnavailableError,
)
from ekip_ratelimit.memory import InMemoryRateLimiter
from fastapi import FastAPI
from idp_testkit import TestIdp

IDP = TestIdp()
URL = "/api/v1/search"


def limits(user: int = 100, tenant: int = 1000, *, fail_open: bool = False) -> RateLimits:
    return RateLimits(
        user=InMemoryRateLimiter(capacity=user, refill_per_second=0.001),
        tenant=InMemoryRateLimiter(capacity=tenant, refill_per_second=0.001),
        fail_open=fail_open,
    )


def client_for(rl: RateLimits) -> httpx.AsyncClient:
    app: FastAPI = create_app(
        Settings(_env_file=None, app_env="test"), token_validator=IDP.validator(), rate_limits=rl
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_user_bucket_allows_burst_then_returns_429_with_retry_after() -> None:
    async with client_for(limits(user=2)) as c:
        first = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
        second = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
        third = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert (first.status_code, second.status_code, third.status_code) == (200, 200, 429)
    assert third.json()["detail"] == "rate limit exceeded"
    assert third.headers["content-type"] == "application/problem+json"
    assert int(third.headers["retry-after"]) >= 1
    assert third.headers["ratelimit-limit"] == "2"
    assert third.headers["ratelimit-remaining"] == "0"
    assert third.headers["cache-control"] == "no-store"


async def test_successful_responses_report_remaining_budget() -> None:
    async with client_for(limits(user=5)) as c:
        r1 = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
        r2 = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert (r1.headers["ratelimit-limit"], r1.headers["ratelimit-remaining"]) == ("5", "4")
    assert r2.headers["ratelimit-remaining"] == "3"


async def test_each_user_has_their_own_bucket() -> None:
    async with client_for(limits(user=1)) as c:
        a1 = await c.get(URL, params={"q": "x"}, headers=IDP.auth(sub="alice"))
        a2 = await c.get(URL, params={"q": "x"}, headers=IDP.auth(sub="alice"))
        b1 = await c.get(URL, params={"q": "x"}, headers=IDP.auth(sub="bob"))
    assert (a1.status_code, a2.status_code, b1.status_code) == (200, 429, 200)


async def test_tenant_bucket_stops_a_noisy_tenant_but_not_others() -> None:
    async with client_for(limits(user=100, tenant=3)) as c:
        codes = [
            (await c.get(URL, params={"q": "x"}, headers=IDP.auth(sub=f"u{i}"))).status_code
            for i in range(5)
        ]
        other_tenant = await c.get(
            URL, params={"q": "x"}, headers=IDP.auth(sub="u0", tenant_id="tenant-other")
        )
    assert codes == [200, 200, 200, 429, 429]  # different users, same tenant, shared budget
    assert other_tenant.status_code == 200


async def test_unauthenticated_and_forbidden_requests_do_not_consume_budget() -> None:
    rl = limits(user=1)
    async with client_for(rl) as c:
        assert (await c.get(URL, params={"q": "x"})).status_code == 401
        assert (await c.get(URL, params={"q": "x"}, headers=IDP.auth(roles=[]))).status_code == 403
        assert (await c.get(URL, params={"q": "x"}, headers=IDP.auth())).status_code == 200


async def test_bucket_keys_contain_tenant_but_never_the_raw_subject() -> None:
    seen: list[str] = []

    class Spy:
        async def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
            seen.append(key)
            return RateLimitDecision(allowed=True, limit=10, remaining=9, retry_after_s=0)

    async with client_for(RateLimits(user=Spy(), tenant=Spy())) as c:
        await c.get(URL, params={"q": "x"}, headers=IDP.auth(sub="very-secret-subject"))
    assert len(seen) == 2
    assert all(k.startswith("tenant-dev:search:") for k in seen)
    assert not any("very-secret-subject" in k for k in seen)


class Down:
    async def check(self, key: str, *, cost: int = 1) -> RateLimitDecision:
        raise RateLimitUnavailableError


async def test_limiter_outage_fails_closed_by_default() -> None:
    async with client_for(RateLimits(user=Down(), tenant=Down())) as c:
        r = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"


async def test_limiter_outage_can_be_configured_to_fail_open() -> None:
    async with client_for(RateLimits(user=Down(), tenant=Down(), fail_open=True)) as c:
        r = await c.get(URL, params={"q": "x"}, headers=IDP.auth())
    assert r.status_code == 200


def test_default_composition_builds_redis_limiters_from_settings() -> None:
    """No injected limits: the real Redis-backed ones are built (lazily; no connection yet)."""
    app = create_app(Settings(_env_file=None, app_env="test"))
    limits_ = app.state.rate_limits
    assert limits_.user._capacity == 20
    assert limits_.tenant._capacity == 200
    assert limits_.fail_open is False
