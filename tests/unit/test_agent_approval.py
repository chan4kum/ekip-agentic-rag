"""Human approval: pause, resume, reject, identity binding, fail-closed. InMemorySaver only."""

import os

import pytest
from ekip_agents import (
    NOT_APPROVED_MESSAGE,
    ApprovalError,
    ApprovalGate,
    RuleBasedRouter,
    RunStatus,
    StateViolation,
    build_agent_graph,
    create_checkpointer,
    pending_approval,
    resume_run,
    start_run,
    thread_config,
)
from ekip_agents.checkpointing import PostgresCheckpointer, to_psycopg_url
from ekip_contracts.conversation import ConversationKey
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_persistence.conversations import InMemoryConversationStore
from fake_llm import FakeChatModel
from langgraph.checkpoint.memory import InMemorySaver

ALICE = SecurityContext(subject="u-1", tenant_id="tenant-a", principal="alice")
BOB = SecurityContext(subject="u-2", tenant_id="tenant-a", principal="bob")
HIT = RetrievedChunk(
    chunk_id="c1", document_id="d1", text="Torque is 140 Nm.", score=1.0, title="Manual"
)
GATE = ApprovalGate("release_answer", "Release this answer to the user?")
Q = "What is the torque value?"


class OneHit:
    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        return [HIT]


def make_graph(store: InMemoryConversationStore | None = None, model: FakeChatModel | None = None):  # type: ignore[no-untyped-def]
    return build_agent_graph(
        router=RuleBasedRouter(),
        retriever=OneHit(),
        model=model or FakeChatModel("Torque is 140 Nm [1]."),
        approval=GATE,
        checkpointer=InMemorySaver(),
        store=store,
    )


async def test_the_run_pauses_for_a_human_and_reports_the_request() -> None:
    graph = make_graph()
    state, pending = await start_run(graph, Q, ALICE, "run-1")
    assert pending is not None and pending.action == "release_answer"
    assert pending.summary == GATE.summary
    assert state["status"] is RunStatus.ANSWERED  # drafted and verified, not yet released
    assert await pending_approval(graph, ALICE, "run-1") == pending


async def test_approval_resumes_and_finishes_the_run() -> None:
    graph = make_graph()
    await start_run(graph, Q, ALICE, "run-1")
    final = await resume_run(graph, ALICE, "run-1", approve=True)
    assert final["status"] is RunStatus.ANSWERED and final["answer"] == "Torque is 140 Nm [1]."
    assert final["approval"].value == "approved"
    assert await pending_approval(graph, ALICE, "run-1") is None


async def test_rejection_ends_refused_and_erases_the_answer() -> None:
    graph = make_graph()
    await start_run(graph, Q, ALICE, "run-1")
    final = await resume_run(graph, ALICE, "run-1", approve=False)
    assert final["status"] is RunStatus.REFUSED and final["reason"] == "approval_rejected"
    assert final["answer"] == NOT_APPROVED_MESSAGE and "140" not in final["answer"]


async def test_a_rejected_run_is_not_remembered() -> None:
    store = InMemoryConversationStore()
    graph = make_graph(store)
    await start_run(graph, Q, ALICE, "run-1")
    await resume_run(graph, ALICE, "run-1", approve=False)
    assert await store.recent(ConversationKey.from_context(ALICE, "conv-0001"), 5) == []


async def test_nothing_is_saved_to_memory_before_approval() -> None:
    store = InMemoryConversationStore()
    graph = make_graph(store)
    await start_run(graph, Q, ALICE, "run-1")
    assert all(not turns for turns in store._threads.values())  # noqa: SLF001


async def test_another_user_cannot_see_or_resume_the_run() -> None:
    graph = make_graph()
    await start_run(graph, Q, ALICE, "run-1")
    assert await pending_approval(graph, BOB, "run-1") is None
    with pytest.raises(ApprovalError):
        await resume_run(graph, BOB, "run-1", approve=True)
    # and the owner's run is still waiting
    assert await pending_approval(graph, ALICE, "run-1") is not None


async def test_resuming_a_finished_or_unknown_run_is_an_error() -> None:
    graph = make_graph()
    with pytest.raises(ApprovalError):
        await resume_run(graph, ALICE, "never", approve=True)
    await start_run(graph, Q, ALICE, "run-1")
    await resume_run(graph, ALICE, "run-1", approve=True)
    with pytest.raises(ApprovalError):
        await resume_run(graph, ALICE, "run-1", approve=True)


async def test_runs_that_do_not_end_answered_never_ask_a_human() -> None:
    graph = make_graph()
    state, pending = await start_run(graph, "thanks!", ALICE, "run-2")
    assert pending is None and state["status"] is RunStatus.REFUSED


def test_an_approval_gate_without_a_checkpointer_is_refused_at_build_time() -> None:
    with pytest.raises(ValueError, match="checkpointer"):
        build_agent_graph(
            router=RuleBasedRouter(), retriever=OneHit(), model=FakeChatModel(), approval=GATE
        )


@pytest.mark.parametrize("key", ["", "a:b", "x" * 65])
def test_bad_thread_keys_are_rejected(key: str) -> None:
    with pytest.raises(StateViolation):
        thread_config(ALICE, key)


def test_thread_ids_contain_tenant_and_user() -> None:
    cfg = thread_config(ALICE, "run-1")
    assert cfg["configurable"]["thread_id"] == "tenant-a:u-1:run-1"


# --- checkpointer factory -----------------------------------------------------------------


def test_the_factory_builds_each_kind() -> None:
    assert create_checkpointer("none") == (None, None)
    saver, owner = create_checkpointer("memory")
    assert isinstance(saver, InMemorySaver) and owner is None
    with pytest.raises(ValueError, match="database url"):
        create_checkpointer("postgres")


def test_the_postgres_url_loses_its_driver_suffix() -> None:
    assert to_psycopg_url("postgresql+asyncpg://u:p@h:1/db") == "postgresql://u:p@h:1/db"
    assert to_psycopg_url("postgresql://u:p@h:1/db") == "postgresql://u:p@h:1/db"


def test_building_a_postgres_checkpointer_does_no_io() -> None:
    saver, owner = create_checkpointer("postgres", "postgresql+asyncpg://u:p@localhost:1/db")
    assert saver is owner.saver  # type: ignore[union-attr]


@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs Postgres")
async def test_a_paused_run_survives_in_postgres() -> None:
    url = os.environ["DATABASE_URL"]
    saver, owner = create_checkpointer("postgres", url)
    assert isinstance(owner, PostgresCheckpointer) and saver is not None
    await owner.open()
    try:
        graph = build_agent_graph(
            router=RuleBasedRouter(),
            retriever=OneHit(),
            model=FakeChatModel("Torque is 140 Nm [1]."),
            approval=GATE,
            checkpointer=saver,
        )
        key = f"it-{os.getpid()}"
        await start_run(graph, Q, ALICE, key)
        assert await pending_approval(graph, ALICE, key) is not None
        final = await resume_run(graph, ALICE, key, approve=True)
        assert final["status"] is RunStatus.ANSWERED
    finally:
        await owner.aclose()


async def test_building_a_postgres_checkpointer_inside_a_running_loop_works_too() -> None:
    saver, owner = create_checkpointer("postgres", "postgresql://u:p@localhost:1/db")
    assert saver is not None and owner is not None
