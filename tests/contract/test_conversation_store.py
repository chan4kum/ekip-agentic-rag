"""One behaviour spec for every ConversationStore. Memory always runs; Postgres with
RUN_INTEGRATION=1 (the migration is applied first)."""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from ekip_contracts.conversation import (
    MAX_TURN_CHARS,
    MAX_TURNS_KEPT,
    ConversationKey,
    ConversationStore,
    Turn,
)
from ekip_contracts.documents import ContractError
from ekip_contracts.identity import SecurityContext
from ekip_persistence import InMemoryConversationStore, PostgresConversationStore, create_pool

LIVE = os.environ.get("RUN_INTEGRATION") == "1"
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://ekip:ekip@localhost:5432/ekip")
CONV = "conversation-0001"


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.integration)])
async def store(request: pytest.FixtureRequest, migrated: None) -> AsyncIterator[ConversationStore]:
    if request.param == "memory":
        yield InMemoryConversationStore()
        return
    if not LIVE:
        pytest.skip("needs live Compose stack (RUN_INTEGRATION=1)")
    pool = await create_pool(DATABASE_URL, max_size=5)
    try:
        yield PostgresConversationStore(pool)
        await pool.execute("DELETE FROM conversation_turns WHERE tenant_id LIKE 'ct-%'")
    finally:
        await pool.close()


@pytest.fixture
def tenant() -> str:
    return f"ct-{uuid.uuid4().hex[:10]}"  # unique per test, so runs never collide


def key(tenant: str, user: str = "alice", conv: str = CONV) -> ConversationKey:
    return ConversationKey(tenant, user, conv)


async def test_turns_come_back_oldest_first_with_a_timestamp(
    store: ConversationStore, tenant: str
) -> None:
    for i in range(3):
        await store.append(key(tenant), Turn(f"q{i}", f"a{i} [1]"))
    turns = await store.recent(key(tenant))
    assert [t.question for t in turns] == ["q0", "q1", "q2"]
    assert all(t.created_at is not None for t in turns)


async def test_recent_returns_only_the_latest_turns(store: ConversationStore, tenant: str) -> None:
    for i in range(5):
        await store.append(key(tenant), Turn(f"q{i}", f"a{i}"))
    assert [t.question for t in await store.recent(key(tenant), limit=2)] == ["q3", "q4"]


async def test_an_unknown_conversation_is_empty(store: ConversationStore, tenant: str) -> None:
    assert await store.recent(key(tenant)) == []


async def test_another_user_with_the_same_conversation_id_sees_nothing(
    store: ConversationStore, tenant: str
) -> None:
    await store.append(key(tenant, "alice"), Turn("secret question", "secret answer"))
    assert await store.recent(key(tenant, "bob")) == []
    assert len(await store.recent(key(tenant, "alice"))) == 1


async def test_another_tenant_with_the_same_user_and_id_sees_nothing(
    store: ConversationStore, tenant: str
) -> None:
    await store.append(key(tenant), Turn("q", "a"))
    assert await store.recent(key(tenant + "x")) == []


async def test_conversations_are_independent(store: ConversationStore, tenant: str) -> None:
    await store.append(key(tenant, conv="conversation-AAAA"), Turn("qa", "aa"))
    await store.append(key(tenant, conv="conversation-BBBB"), Turn("qb", "ab"))
    assert [t.question for t in await store.recent(key(tenant, conv="conversation-AAAA"))] == ["qa"]


async def test_old_turns_are_dropped_beyond_the_cap(store: ConversationStore, tenant: str) -> None:
    for i in range(MAX_TURNS_KEPT + 5):
        await store.append(key(tenant), Turn(f"q{i}", "a"))
    turns = await store.recent(key(tenant), limit=50)
    assert len(turns) == MAX_TURNS_KEPT
    assert turns[0].question == "q5" and turns[-1].question == f"q{MAX_TURNS_KEPT + 4}"


async def test_delete_removes_only_that_conversation(store: ConversationStore, tenant: str) -> None:
    await store.append(key(tenant), Turn("q1", "a1"))
    await store.append(key(tenant), Turn("q2", "a2"))
    await store.append(key(tenant, "bob"), Turn("bq", "ba"))
    assert await store.delete(key(tenant)) == 2
    assert await store.recent(key(tenant)) == []
    assert len(await store.recent(key(tenant, "bob"))) == 1
    assert await store.delete(key(tenant)) == 0


@pytest.mark.parametrize("limit", [0, 51, -1])
async def test_bad_limits_are_rejected(store: ConversationStore, tenant: str, limit: int) -> None:
    with pytest.raises(ContractError):
        await store.recent(key(tenant), limit=limit)


async def test_special_characters_are_stored_verbatim(
    store: ConversationStore, tenant: str
) -> None:
    text = "'; DROP TABLE conversation_turns; -- é中 <b>x</b>"
    await store.append(key(tenant), Turn(text, text))
    [turn] = await store.recent(key(tenant))
    assert turn.question == text and turn.answer == text


# ---- contract objects (no store needed) -------------------------------------------------------


@pytest.mark.parametrize(
    ("tenant", "user", "conv"),
    [("", "u", CONV), ("t/x", "u", CONV), ("t", "", CONV), ("t", "has space", CONV),
     ("t", "u", "short"), ("t", "u", "bad id with spaces!")],
)  # fmt: skip
def test_invalid_keys_are_rejected(tenant: str, user: str, conv: str) -> None:
    with pytest.raises(ContractError):
        ConversationKey(tenant, user, conv)


@pytest.mark.parametrize("q, a", [("", "a"), ("q", "  "), ("x" * (MAX_TURN_CHARS + 1), "a")])
def test_invalid_turns_are_rejected(q: str, a: str) -> None:
    with pytest.raises(ContractError):
        Turn(q, a)


def test_the_key_is_built_from_the_verified_identity() -> None:
    ctx = SecurityContext(subject="user-42", tenant_id="tenant-a", principal="alice")
    assert ConversationKey.from_context(ctx, CONV) == ConversationKey("tenant-a", "user-42", CONV)
