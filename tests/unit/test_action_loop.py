from collections.abc import Mapping, Sequence
from typing import Any

from ekip_agents.action import (
    ActionLimits,
    ActionStatus,
    ActionStep,
    Finish,
    Observation,
    run_action_loop,
    run_action_node,
)
from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ApprovalRequest, ToolResult, ToolStatus
from ekip_tools import arguments_digest
from tools_testkit import ALICE, BOB_ADMIN, Harness, make_spec, write_spec


class ScriptedPlanner:
    def __init__(self, *decisions: ActionStep | Finish) -> None:
        self._decisions = list(decisions)
        self.seen: list[tuple[Observation, ...]] = []

    async def next(self, question: str, observations: Sequence[Observation]) -> ActionStep | Finish:
        self.seen.append(tuple(observations))
        if not self._decisions:
            return Finish(answer="done")
        return self._decisions.pop(0)


class FakeInvoker:
    def __init__(self, status: ToolStatus = ToolStatus.OK) -> None:
        self.status = status
        self.calls: list[tuple[str, Mapping[str, Any], str | None]] = []

    async def call(
        self,
        context: SecurityContext,
        name: str,
        arguments: Mapping[str, Any],
        *,
        approval: str | None = None,
    ) -> ToolResult:
        self.calls.append((name, arguments, approval))
        return ToolResult(
            tool=name, status=self.status, content=f"<tool_output>{name}</tool_output>"
        )


def step(text: str = "a", tool: str = "echo_tool") -> ActionStep:
    return ActionStep(tool=tool, arguments={"text": text})


async def test_runs_tools_then_finishes_with_observations_visible_to_the_planner() -> None:
    planner = ScriptedPlanner(step("a"), step("b"), Finish(answer="final"))
    invoker = FakeInvoker()
    outcome = await run_action_loop(invoker, ALICE, planner, "q")
    assert outcome.status is ActionStatus.COMPLETED
    assert outcome.answer == "final"
    assert outcome.tool_calls_used == 2
    assert len(planner.seen[-1]) == 2  # the planner saw both observations


async def test_step_and_tool_call_budgets_end_the_run() -> None:
    steps = [step(str(i)) for i in range(20)]
    out = await run_action_loop(
        FakeInvoker(),
        ALICE,
        ScriptedPlanner(*steps),
        "q",
        limits=ActionLimits(max_steps=3, max_tool_calls=10),
    )
    assert (out.status, out.reason) == (ActionStatus.LIMIT_REACHED, "max_steps")
    out = await run_action_loop(
        FakeInvoker(),
        ALICE,
        ScriptedPlanner(*steps),
        "q",
        limits=ActionLimits(max_steps=10, max_tool_calls=2),
    )
    assert (out.status, out.reason) == (ActionStatus.LIMIT_REACHED, "max_tool_calls")
    assert out.tool_calls_used == 2


async def test_repeating_the_same_call_stops_the_loop() -> None:
    invoker = FakeInvoker()
    out = await run_action_loop(invoker, ALICE, ScriptedPlanner(step("x"), step("x")), "q")
    assert (out.status, out.reason) == (ActionStatus.LIMIT_REACHED, "repeated_call")
    assert len(invoker.calls) == 1


async def test_refusals_are_counted_and_stop_the_loop() -> None:
    invoker = FakeInvoker(ToolStatus.DENIED)
    out = await run_action_loop(
        invoker,
        ALICE,
        ScriptedPlanner(step("a"), step("b"), step("c")),
        "q",
        limits=ActionLimits(max_refusals=2),
    )
    assert (out.status, out.reason) == (ActionStatus.LIMIT_REACHED, "too_many_refusals")
    assert len(invoker.calls) == 2


async def test_planner_failure_becomes_a_failed_outcome() -> None:
    class Broken:
        async def next(
            self, question: str, observations: Sequence[Observation]
        ) -> ActionStep | Finish:
            raise RuntimeError("model exploded: api_key=sk-123")

    out = await run_action_loop(FakeInvoker(), ALICE, Broken(), "q")
    assert out.status is ActionStatus.FAILED
    assert "sk-123" not in out.model_dump_json()


async def test_write_step_stops_for_approval_and_resumes_with_the_grant() -> None:
    h = Harness(
        [
            write_spec(
                "create_ticket",
                input_schema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string", "maxLength": 50}},
                },
            )
        ]
    )
    write = step("page the on-call", tool="create_ticket")
    planner = ScriptedPlanner(write, Finish(answer="ticket opened"))
    stopped = await run_action_loop(h.gateway, ALICE, planner, "q")
    assert stopped.status is ActionStatus.NEEDS_APPROVAL
    assert stopped.pending is not None
    assert stopped.pending.request.arguments_digest == arguments_digest(write.arguments)
    assert h.backend.calls == []

    grant = h.approvals.issue(stopped.pending.request, BOB_ADMIN)
    resumed = await run_action_loop(
        h.gateway, ALICE, planner, "q", resume=stopped.pending, grant=grant
    )
    assert resumed.status is ActionStatus.COMPLETED
    assert resumed.answer == "ticket opened"
    assert len(h.backend.calls) == 1


async def test_a_wrong_grant_on_resume_cannot_run_the_write() -> None:
    h = Harness([write_spec("create_ticket")])
    write = step("hello", tool="create_ticket")
    stopped = await run_action_loop(h.gateway, ALICE, ScriptedPlanner(write), "q")
    assert stopped.pending is not None
    other = ActionStep(tool="create_ticket", arguments={"text": "different"})
    other_request = h.approvals.request(ALICE, make_spec("create_ticket"), other.arguments)
    wrong_grant = h.approvals.issue(other_request, BOB_ADMIN)
    out = await run_action_loop(
        h.gateway, ALICE, ScriptedPlanner(), "q", resume=stopped.pending, grant=wrong_grant
    )
    assert h.backend.calls == []
    assert out.status in {ActionStatus.NEEDS_APPROVAL, ActionStatus.COMPLETED}


async def test_graph_node_returns_plain_data_and_never_touches_protected_keys() -> None:
    state = {"security": ALICE, "question": "q"}
    update = await run_action_node(
        state, invoker=FakeInvoker(), planner=ScriptedPlanner(step("a"), Finish(answer="ok"))
    )
    assert update["action_status"] == "completed"
    assert update["answer"] == "ok"
    assert not {"security", "question"} & set(update)
    assert isinstance(update["action_outcome"], dict)


async def test_graph_node_exposes_the_pending_approval_for_checkpointing() -> None:
    request = ApprovalRequest(
        request_id="r1",
        tool="create_ticket",
        arguments_digest="d",
        requested_by="u-alice",
        tenant_id="tenant-a",
        summary="s",
        expires_at=1,
    )

    class Asks(FakeInvoker):
        async def call(
            self,
            context: SecurityContext,
            name: str,
            arguments: Mapping[str, Any],
            *,
            approval: str | None = None,
        ) -> ToolResult:
            return ToolResult(
                tool=name, status=ToolStatus.APPROVAL_REQUIRED, approval_request=request
            )

    update = await run_action_node(
        {"security": ALICE, "question": "q"},
        invoker=Asks(),
        planner=ScriptedPlanner(step("a", "create_ticket")),
    )
    assert update["action_status"] == "needs_approval"
    assert update["action_resume"]["request"]["request_id"] == "r1"
