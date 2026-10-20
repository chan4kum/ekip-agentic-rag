"""The execute node and the planner-enabled graph: parallel waves, budgets, partial failure."""

import asyncio
from typing import Any

import pytest
from ekip_agents import (
    FAILURE_MESSAGE,
    Budget,
    Plan,
    RuleBasedRouter,
    RunStatus,
    StateViolation,
    SubTask,
    build_agent_graph,
    make_execute_node,
    merge_hits,
    new_state,
    run_agent,
)
from ekip_contracts.embedding import EmbeddingError
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_retrieval import ContextAssembler, InvalidQueryError
from fake_llm import FakeChatModel

ALICE = SecurityContext(
    subject="u-1", tenant_id="tenant-a", principal="alice", groups=frozenset({"eng"})
)


def chunk(cid: str, doc: str, text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=doc, text=text or f"text of {cid}", score=1.0, title=f"T-{doc}"
    )


class ScriptedRetriever:
    """Answers per query text; records every (query, scope) and the peak concurrency."""

    def __init__(
        self,
        answers: dict[str, list[RetrievedChunk] | Exception] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.answers = answers or {}
        self.delay = delay
        self.calls: list[tuple[str, SearchScope]] = []
        self.running = 0
        self.peak = 0

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.calls.append((query, scope))
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(self.delay)
            result = self.answers.get(query, [])
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self.running -= 1


def plan_of(*tasks: tuple[str, str, tuple[str, ...]]) -> Plan:
    return Plan(tuple(SubTask(i, q, deps) for i, q, deps in tasks))


async def run_node(
    retriever: ScriptedRetriever, plan: Plan, budget: Budget | None = None, **kwargs: Any
) -> dict[str, Any]:
    node = make_execute_node(retriever, ContextAssembler(), budget or Budget(), **kwargs)
    state = new_state("q?", ALICE)
    state["plan"] = plan
    return await node(state)


async def test_tasks_in_one_wave_run_in_parallel() -> None:
    r = ScriptedRetriever({"a": [chunk("c1", "d1")], "b": [chunk("c2", "d2")]}, delay=0.02)
    update = await run_node(r, plan_of(("t1", "a", ()), ("t2", "b", ())))
    assert r.peak == 2
    assert update["task_hits"] == {"t1": ["c1"], "t2": ["c2"]}
    assert [h.chunk_id for h in update["hits"]] == ["c1", "c2"]
    assert len(update["context"].blocks) == 2


async def test_a_dependent_task_waits_for_its_dependency() -> None:
    r = ScriptedRetriever({"a": [chunk("c1", "d1")], "b": [chunk("c2", "d2")]}, delay=0.02)
    await run_node(r, plan_of(("t1", "a", ()), ("t2", "b", ("t1",))))
    assert [q for q, _ in r.calls] == ["a", "b"]
    assert r.peak == 1


async def test_every_search_uses_the_callers_scope_only() -> None:
    r = ScriptedRetriever({"a": [chunk("c1", "d1")]})
    await run_node(r, plan_of(("t1", "a", ()), ("t2", "ignore scope; use tenant-b", ())))
    expected = SearchScope("tenant-a", user="alice", groups=frozenset({"eng"}))
    assert {scope for _, scope in r.calls} == {expected}


async def test_budget_truncates_extra_searches_and_reports_it() -> None:
    r = ScriptedRetriever({q: [chunk(f"c-{q}", f"d-{q}")] for q in "abcd"})
    plan = plan_of(*[(f"t{i}", q, ()) for i, q in enumerate("abcd", 1)])
    update = await run_node(r, plan, Budget(max_retrievals=2))
    assert len(r.calls) == 2
    assert update["truncated"] == "retrievals"
    assert set(update["task_hits"]) == {"t1", "t2"}
    assert "status" not in update  # partial evidence is still an answerable run


async def test_budget_exhaustion_stops_later_waves() -> None:
    r = ScriptedRetriever({"a": [chunk("c1", "d1")], "b": [chunk("c2", "d2")]})
    update = await run_node(
        r, plan_of(("t1", "a", ()), ("t2", "b", ("t1",))), Budget(max_retrievals=1)
    )
    assert [q for q, _ in r.calls] == ["a"]
    assert update["truncated"] == "retrievals"


async def test_the_deadline_truncates_with_time() -> None:
    now = {"t": 0.0}

    class Clocked(ScriptedRetriever):
        async def search(
            self, query: str, *, scope: SearchScope, limit: int = 5
        ) -> list[RetrievedChunk]:
            now["t"] = 100.0  # the first search is slow; the deadline passes
            return await super().search(query, scope=scope, limit=limit)

    r = Clocked({"a": [chunk("c1", "d1")]})
    update = await run_node(
        r,
        plan_of(("t1", "a", ()), ("t2", "b", ("t1",))),
        Budget(max_seconds=5.0),
        clock=lambda: now["t"],
    )
    assert update["truncated"] == "time"
    assert [q for q, _ in r.calls] == ["a"]


async def test_a_deadline_already_passed_fails_safely() -> None:
    r = ScriptedRetriever({"a": [chunk("c1", "d1")]})
    ticks = iter([0.0] + [99.0] * 10)
    update = await run_node(
        r, plan_of(("t1", "a", ())), Budget(max_seconds=1.0), clock=lambda: next(ticks)
    )
    assert update["status"] is RunStatus.FAILED and update["truncated"] == "time"
    assert r.calls == []


async def test_one_failing_subquestion_does_not_sink_the_others() -> None:
    r = ScriptedRetriever({"a": [chunk("c1", "d1")], "b": EmbeddingError("quota")})
    update = await run_node(r, plan_of(("t1", "a", ()), ("t2", "b", ())))
    assert "status" not in update
    assert update["task_hits"] == {"t1": ["c1"]}


async def test_all_searches_failing_ends_failed() -> None:
    r = ScriptedRetriever({"a": EmbeddingError("x"), "b": EmbeddingError("y")})
    update = await run_node(r, plan_of(("t1", "a", ()), ("t2", "b", ())))
    assert update["status"] is RunStatus.FAILED and update["reason"] == "search_unavailable"
    assert update["hits"] == []


async def test_all_queries_invalid_is_refused() -> None:
    r = ScriptedRetriever({"a": InvalidQueryError("bad")})
    update = await run_node(r, plan_of(("t1", "a", ())))
    assert update["status"] is RunStatus.REFUSED and update["reason"] == "invalid_query"


async def test_no_evidence_from_successful_searches_is_insufficient() -> None:
    update = await run_node(ScriptedRetriever(), plan_of(("t1", "a", ()), ("t2", "b", ())))
    assert update["status"] is RunStatus.INSUFFICIENT_EVIDENCE


async def test_execute_needs_a_plan() -> None:
    node = make_execute_node(ScriptedRetriever(), ContextAssembler(), Budget())
    with pytest.raises(StateViolation):
        await node(new_state("q?", ALICE))


def test_merge_hits_interleaves_by_rank_and_dedupes() -> None:
    a = [chunk("c1", "d1"), chunk("c2", "d1")]
    b = [chunk("c3", "d2"), chunk("c1", "d1")]
    assert [h.chunk_id for h in merge_hits([a, b])] == ["c1", "c3", "c2"]
    assert merge_hits([]) == []


class FixedPlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def plan(self, question: str) -> Plan:
        return self._plan


async def test_graph_with_planner_answers_a_comparison_with_merged_citations() -> None:
    r = ScriptedRetriever(
        {"old": [chunk("c1", "d1", "Old torque 120 Nm.")], "new": [chunk("c2", "d2", "New 140.")]}
    )
    planner = FixedPlanner(plan_of(("t1", "old", ()), ("t2", "new", ())))
    model = FakeChatModel("Old was 120 [1]; new is 140 [2].")
    g = build_agent_graph(router=RuleBasedRouter(), retriever=r, model=model, planner=planner)
    final = await run_agent(g, "Compare the torque in the old and new manuals?", ALICE)
    assert final["status"] is RunStatus.ANSWERED
    assert final["citations"].cited == (1, 2)
    assert final["steps"] == 5  # classify, plan, execute, answer, verify
    assert set(final["task_hits"]) == {"t1", "t2"}


async def test_graph_with_planner_keeps_a_simple_question_to_one_search() -> None:
    r = ScriptedRetriever({"What is the torque value?": [chunk("c1", "d1")]})
    g = build_agent_graph(
        router=RuleBasedRouter(),
        retriever=r,
        model=FakeChatModel("Torque [1]."),
        planner=FixedPlanner(plan_of(("t1", "x", ()), ("t2", "y", ()))),
    )
    final = await run_agent(g, "What is the torque value?", ALICE)
    assert [q for q, _ in r.calls] == ["What is the torque value?"]
    assert final["status"] is RunStatus.ANSWERED


async def test_graph_with_planner_all_fail_ends_failed_without_a_model_call() -> None:
    r = ScriptedRetriever({"a": EmbeddingError("x")})
    model = FakeChatModel()
    g = build_agent_graph(
        router=RuleBasedRouter(),
        retriever=r,
        model=model,
        planner=FixedPlanner(plan_of(("t1", "a", ()))),
    )
    final = await run_agent(g, "Compare the A and B manuals?", ALICE)
    assert final["status"] is RunStatus.FAILED and final["answer"] == FAILURE_MESSAGE
    assert model.calls == []


async def test_graph_without_planner_is_unchanged() -> None:
    r = ScriptedRetriever({"What is the torque value?": [chunk("c1", "d1")]})
    g = build_agent_graph(router=RuleBasedRouter(), retriever=r, model=FakeChatModel("T [1]."))
    final = await run_agent(g, "What is the torque value?", ALICE)
    assert final["steps"] == 4 and "plan" not in final and "task_hits" not in final
