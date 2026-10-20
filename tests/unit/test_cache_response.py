import asyncio
import json
from collections.abc import Awaitable, Callable

import pytest
from cache_testkit import VERSIONS, Counter, Stack, answer, make_stack, scope
from ekip_caching import CacheVersions, scope_fingerprint
from ekip_contracts.cache import CachedAnswer
from ekip_contracts.retrieval import SearchScope

Q = "What is the torque value?"


async def test_second_identical_request_is_a_hit() -> None:
    s = make_stack()
    work = Counter(answer("d1", "d2"))
    first, hit1 = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    second, hit2 = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert (hit1, hit2, work.calls) == (False, True, 1)
    assert second == first
    assert s.metrics.get("response", "hit") == 1 and s.metrics.get("response", "miss") == 1


async def test_cosmetic_query_differences_share_an_entry() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    _, hit = await s.response.get_or_compute(scope(), f"  {Q.upper()} ", work, authz_version="a1")
    assert hit and work.calls == 1


async def test_cross_tenant_never_hits() -> None:
    s = make_stack()
    a = Counter(answer("d1", text="acme answer"))
    b = Counter(answer("d9", text="globex answer"))
    await s.response.get_or_compute(scope("acme"), Q, a, authz_version="a1")
    got, hit = await s.response.get_or_compute(scope("globex"), Q, b, authz_version="a1")
    assert not hit and got.answer == "globex answer" and b.calls == 1


async def test_different_groups_or_user_never_reuse_each_others_entry() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope("acme", "alice", "hr"), Q, work, authz_version="a1")
    for other in (
        scope("acme", "alice", "finance"),
        scope("acme", "alice", "hr", "finance"),
        scope("acme", "bob", "hr"),
        scope("acme", "alice"),
    ):
        _, hit = await s.response.get_or_compute(other, Q, work, authz_version="a1")
        assert not hit
    assert work.calls == 5


async def test_same_identity_hits_regardless_of_group_order() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope("acme", "a", "x", "y"), Q, work, authz_version="a1")
    _, hit = await s.response.get_or_compute(
        scope("acme", "a", "y", "x"), Q, work, authz_version="a1"
    )
    assert hit


async def test_revoked_access_is_stale_and_entry_is_dropped() -> None:
    s = make_stack()
    work = Counter(answer("d1", "d2"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    s.visibility.hidden.add("d2")  # alice lost access to d2 (no invalidation event yet)
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit and work.calls == 2
    assert s.metrics.get("response", "stale") == 1
    # the recomputed answer is cached again, and the checker keeps being consulted on hits
    s.visibility.hidden.clear()
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert hit


async def test_visibility_is_checked_against_every_source_document() -> None:
    s = make_stack()
    work = Counter(answer("d1", "d2", "d3"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert s.visibility.calls[-1][1] == {"d1", "d2", "d3"}
    assert s.visibility.calls[-1][0] == scope()


async def test_deletion_makes_cached_answers_unreachable_even_if_visibility_says_yes() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    await s.response.get_or_compute(scope("acme", "bob", "g"), Q, work, authz_version="a1")
    await s.invalidator.document_deleted("acme", "d1")
    for who in (scope(), scope("acme", "bob", "g")):
        _, hit = await s.response.get_or_compute(who, Q, work, authz_version="a1")
        assert not hit
    assert work.calls == 4


async def test_acl_change_in_one_tenant_leaves_other_tenants_cached() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope("globex"), Q, work, authz_version="a1")
    await s.invalidator.acl_changed("acme", "d1")
    _, hit = await s.response.get_or_compute(scope("globex"), Q, work, authz_version="a1")
    assert hit


async def test_visibility_failure_fails_closed_and_recomputes() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    s.visibility.broken = True
    got, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit and work.calls == 2 and got.answer
    assert s.metrics.get("response", "blocked") == 1


async def test_unexpected_checker_exception_also_fails_closed() -> None:
    s = make_stack()

    class Boom:
        async def all_visible(self, scope: SearchScope, document_ids: object) -> bool:
            raise RuntimeError("bug")

    s.response._core._visibility = Boom()  # noqa: SLF001
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit and work.calls == 2


@pytest.mark.parametrize(
    "changed",
    [
        {"retrieval_config": "rc2"},
        {"prompt": "answer-v3"},
        {"model": "gpt-y"},
        {"output_policy": "op2"},
    ],
)
async def test_any_version_change_is_a_miss(changed: dict[str, str]) -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    v2 = CacheVersions(
        **{
            "retrieval_config": VERSIONS.retrieval_config,
            "prompt": VERSIONS.prompt,
            "model": VERSIONS.model,
            "output_policy": VERSIONS.output_policy,
            **changed,
        }
    )
    other = make_stack(versions=v2)
    other.store._data = s.store._data  # noqa: SLF001
    other.store._counters = s.store._counters  # noqa: SLF001
    _, hit = await other.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit and work.calls == 2


async def test_authorization_scope_version_change_is_a_miss() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a2")
    assert not hit


async def test_ttl_expiry() -> None:
    s = make_stack(ttl_s=60)
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    s.clock.now += 59
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert hit
    s.clock.now += 2
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit and work.calls == 2


async def test_stampede_runs_one_computation() -> None:
    s = make_stack()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow() -> CachedAnswer:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return answer("d1")

    tasks = [
        asyncio.create_task(s.response.get_or_compute(scope(), Q, slow, authz_version="a1"))
        for _ in range(20)
    ]
    await started.wait()
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*tasks)
    assert calls == 1 and all(r[0].answer for r in results)


async def test_stampede_does_not_merge_different_identities() -> None:
    s = make_stack()
    release = asyncio.Event()
    seen: list[str] = []

    def make(who: str) -> Callable[[], Awaitable[CachedAnswer]]:
        async def work() -> CachedAnswer:
            seen.append(who)
            await release.wait()
            return answer("d1", text=f"for {who}")

        return work

    t1 = asyncio.create_task(
        s.response.get_or_compute(scope("acme", "alice"), Q, make("alice"), authz_version="a")
    )
    t2 = asyncio.create_task(
        s.response.get_or_compute(scope("acme", "bob"), Q, make("bob"), authz_version="a")
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    release.set()
    (a, _), (b, _) = await asyncio.gather(t1, t2)
    assert sorted(seen) == ["alice", "bob"]
    assert (a.answer, b.answer) == ("for alice", "for bob")


async def test_compute_error_is_not_cached_and_propagates() -> None:
    s = make_stack()

    async def boom() -> CachedAnswer:
        raise RuntimeError("model down")

    with pytest.raises(RuntimeError):
        await s.response.get_or_compute(scope(), Q, boom, authz_version="a1")
    work = Counter(answer("d1"))
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit and work.calls == 1


async def _only_key(s: Stack) -> str:
    keys = [k for k in s.store._data if ":response:" in k]  # noqa: SLF001
    assert len(keys) == 1
    return keys[0]


async def test_corrupted_value_is_a_miss_not_a_crash() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    key = await _only_key(s)
    for junk in (b"\xff\xfe not json", b"[]", b'{"v":1}', b"null", b'{"v":9,"layer":"response"}'):
        await s.store.set(key, junk, ttl_s=60)
        got, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
        assert not hit and got.answer
    assert s.metrics.get("response", "corrupt") == 5


async def test_schema_invalid_payload_is_a_miss() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    key = await _only_key(s)
    envelope = json.loads(await s.store.get(key) or b"")
    envelope["data"]["answer"] = ""  # violates min_length
    await s.store.set(key, json.dumps(envelope).encode(), ttl_s=60)
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit


async def test_oversized_value_is_not_served_and_not_stored() -> None:
    s = make_stack(max_bytes=2048)
    big = Counter(answer("d1", text="x" * 5000))
    _, hit = await s.response.get_or_compute(scope(), Q, big, authz_version="a1")
    _, hit2 = await s.response.get_or_compute(scope(), Q, big, authz_version="a1")
    assert not hit and not hit2 and big.calls == 2
    assert s.metrics.get("response", "skipped") == 2
    # a value that grew past the limit after being stored is ignored too
    small = Counter(answer("d1"))
    await s.response.get_or_compute(scope(), Q + " other", small, authz_version="a1")
    key = next(k for k in s.store._data if ":response:" in k)  # noqa: SLF001
    await s.store.set(key, b"x" * 4096, ttl_s=60)
    _, hit3 = await s.response.get_or_compute(scope(), Q + " other", small, authz_version="a1")
    assert not hit3


async def test_entry_bound_to_another_caller_is_rejected_even_under_the_right_key() -> None:
    """Defence in depth: if a bug or tampering put alice's bytes under bob's key, bob misses."""
    s = make_stack()
    alice = Counter(answer("d1", text="alice only"))
    await s.response.get_or_compute(scope("acme", "alice"), Q, alice, authz_version="a1")
    alice_key = await _only_key(s)
    bob = Counter(answer("d1", text="bob"))
    await s.response.get_or_compute(scope("acme", "bob"), Q, bob, authz_version="a1")
    bob_key = next(k for k in s.store._data if k != alice_key and ":response:" in k)  # noqa: SLF001
    alice_blob = await s.store.get(alice_key)
    assert alice_blob is not None
    await s.store.set(bob_key, alice_blob, ttl_s=60)
    got, hit = await s.response.get_or_compute(scope("acme", "bob"), Q, bob, authz_version="a1")
    assert not hit and got.answer == "bob"
    assert scope_fingerprint(scope("acme", "alice")) != scope_fingerprint(scope("acme", "bob"))


async def test_ungrounded_answers_are_not_cached() -> None:
    s = make_stack()
    work = Counter(CachedAnswer(answer="I could not find evidence."))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit and work.calls == 2


async def test_caller_without_acl_identity_is_never_cached() -> None:
    s = make_stack()
    work = Counter(answer("d1"))
    anon = SearchScope(tenant_id="acme")
    await s.response.get_or_compute(anon, Q, work, authz_version="a1")
    _, hit = await s.response.get_or_compute(anon, Q, work, authz_version="a1")
    assert not hit and work.calls == 2 and not s.store._data  # noqa: SLF001


async def test_answer_with_too_many_documents_is_skipped() -> None:
    s = make_stack()
    s.response._core._max_tags = 2  # noqa: SLF001
    work = Counter(answer("d1", "d2", "d3"))
    await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    _, hit = await s.response.get_or_compute(scope(), Q, work, authz_version="a1")
    assert not hit


def test_cited_documents_must_be_listed_as_sources() -> None:
    from ekip_contracts.cache import CachedCitation

    with pytest.raises(ValueError, match="source_document_ids"):
        CachedAnswer(answer="x", citations=(CachedCitation(chunk_id="c", document_id="d1"),))
