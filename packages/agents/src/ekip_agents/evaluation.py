"""Compare agent variants (single search vs multi-step) on a versioned dataset.

Same documents/queries format as the retrieval dataset, plus per query:
- `expect`: "answered" | "insufficient_evidence" | "refused" (what a correct system would do)
- `plan`: the sub-questions an offline planner proposes (the real planner is used with --live)

Measured per variant: cited-source recall (did the answer cite the passages that answer the
question?), outcome counts, cost (model calls, searches), and LEAKS - any retrieved or cited
passage the caller may not read, judged from the dataset itself rather than from the system under
test. A single leak is a security bug, so it fails the run.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import RetrievedChunk, Retriever, SearchScope
from ekip_retrieval.evaluation import EvalDataset, EvalQuery, load_dataset, readable_keys

from ekip_agents.graph import Graph, run_agent
from ekip_agents.plan import Plan, SubTask, single_task_plan
from ekip_agents.state import RunStatus

EXPECTATIONS = frozenset({"answered", "insufficient_evidence", "refused"})


@dataclass(frozen=True)
class AgentCase:
    query: EvalQuery
    expect: str
    plan: Plan | None  # offline planner script; None -> a one-task plan


def load_agent_dataset(path: Path) -> tuple[EvalDataset, tuple[AgentCase, ...]]:
    dataset = load_dataset(path)
    extras = {q["id"]: q for q in json.loads(path.read_text())["queries"]}
    cases: list[AgentCase] = []
    for q in dataset.queries:
        raw = extras[q.id]
        expect = raw.get("expect", "answered")
        if expect not in EXPECTATIONS:
            raise ValueError(f"query {q.id}: unknown expectation {expect!r}")
        cases.append(AgentCase(q, expect, _plan(raw.get("plan"))))
    return dataset, tuple(cases)


def _plan(raw: Any) -> Plan | None:
    if raw is None:
        return None
    return Plan(tuple(SubTask(t["id"], t["question"], tuple(t.get("depends_on", ()))) for t in raw))


class ScriptedPlanner:
    """Offline stand-in for the LLM planner: returns the dataset's plan for a known question."""

    def __init__(self, cases: Sequence[AgentCase]) -> None:
        self._plans = {c.query.text: c.plan for c in cases if c.plan is not None}

    async def plan(self, question: str) -> Plan:
        return self._plans.get(question) or single_task_plan(question)


class CountingRetriever:
    """Wraps a retriever to count searches per run (the cost of a variant)."""

    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self.calls = 0

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.calls += 1
        return await self._inner.search(query, scope=scope, limit=limit)


@dataclass(frozen=True)
class AgentEvalReport:
    name: str
    queries: int
    recall: float  # mean cited-source recall over queries that have a readable answer
    recall_by_kind: Mapping[str, float]
    outcomes: Mapping[str, int]  # RunStatus value -> count
    mean_model_calls: float
    mean_searches: float
    truncated: int  # runs cut short by a budget
    leaks: tuple[str, ...]  # "query_id -> passage key"
    unexpected: tuple[str, ...]  # "query_id: expected X, got Y" (informational)
    errors: tuple[str, ...] = field(default=())


def _security(q: EvalQuery) -> SecurityContext:
    return SecurityContext(
        subject=q.scope.user or "eval-user",
        tenant_id=q.scope.tenant_id,
        principal=q.scope.user or "eval-user",
        groups=q.scope.groups,
    )


async def evaluate_agent(
    name: str,
    graph: Graph,
    counter: CountingRetriever,
    dataset: EvalDataset,
    cases: Sequence[AgentCase],
    chunk_to_key: Mapping[str, str],
) -> AgentEvalReport:
    recalls: dict[str, list[float]] = {}
    outcomes: Counter[str] = Counter()
    leaks: list[str] = []
    unexpected: list[str] = []
    errors: list[str] = []
    model_calls = searches = truncated = ran = 0
    for case in cases:
        q = case.query
        before = counter.calls
        try:
            final = await run_agent(graph, q.text, _security(q))
        except Exception as exc:  # a crashing run is reported, never skipped
            errors.append(f"{q.id}: {type(exc).__name__}")
            continue
        ran += 1
        searches += counter.calls - before
        model_calls += final.get("model_calls", 0)
        truncated += 1 if final.get("truncated") else 0
        status = final["status"]
        outcomes[status.value] += 1
        if status.value != case.expect:
            unexpected.append(f"{q.id}: expected {case.expect}, got {status.value}")
        allowed = readable_keys(dataset, q.scope)
        context = final.get("context")
        cited_keys: set[str] = set()
        if context is not None and status is RunStatus.ANSWERED:
            cited_keys = {
                chunk_to_key.get(c.chunk_id, f"?{c.chunk_id}")
                for n in final["citations"].cited
                if (c := context.citation(n)) is not None
            }
        seen = {chunk_to_key.get(h.chunk_id, f"?{h.chunk_id}") for h in final.get("hits", [])}
        leaks.extend(f"{q.id} -> {key}" for key in sorted((seen | cited_keys) - allowed))
        if q.relevant:
            recalls.setdefault(q.kind, []).append(len(cited_keys & q.relevant) / len(q.relevant))
    every = [r for rs in recalls.values() for r in rs]
    return AgentEvalReport(
        name=name,
        queries=ran,
        recall=sum(every) / len(every) if every else 0.0,
        recall_by_kind={k: sum(v) / len(v) for k, v in sorted(recalls.items())},
        outcomes=dict(sorted(outcomes.items())),
        mean_model_calls=model_calls / ran if ran else 0.0,
        mean_searches=searches / ran if ran else 0.0,
        truncated=truncated,
        leaks=tuple(leaks),
        unexpected=tuple(unexpected),
        errors=tuple(errors),
    )


def format_agent_reports(reports: Sequence[AgentEvalReport]) -> str:
    head = (
        "| variant | runs | cited recall | answered | no evidence | refused | failed "
        "| model calls | searches | cut short | leaks | errors |"
    )
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        o = r.outcomes
        lines.append(
            f"| {r.name} | {r.queries} | {r.recall:.3f} | {o.get('answered', 0)} | "
            f"{o.get('insufficient_evidence', 0)} | {o.get('refused', 0)} | {o.get('failed', 0)} | "
            f"{r.mean_model_calls:.1f} | {r.mean_searches:.1f} | {r.truncated} | "
            f"{len(r.leaks)} | {len(r.errors)} |"
        )
    return "\n".join(lines)
