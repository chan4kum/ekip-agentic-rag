"""The bounded action loop: a planner proposes tool calls, the gateway runs them.

The loop is the safety wrapper around an LLM that can act. It guarantees:
- a hard cap on steps and on tool calls, so it always terminates;
- the model only ever proposes a tool NAME and ARGUMENTS; the gateway authorizes and validates;
- a write tool stops the loop with NEEDS_APPROVAL (the graph checkpoints and asks a human); it
  resumes with a signed grant that is bound to the exact arguments;
- repeating the same call, or collecting too many refusals, ends the run (no spinning);
- tool output reaches the planner only as the gateway's wrapped, untrusted text.
Nothing here imports the gateway: it depends on the `ToolInvoker` port in ekip_contracts.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ApprovalRequest, ToolInvoker, ToolResult, ToolStatus
from pydantic import BaseModel, ConfigDict, Field


class ActionStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str = Field(max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        blob = json.dumps([self.tool, self.arguments], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()


class Finish(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str = Field(max_length=8000)


class Observation(BaseModel):
    """What the planner sees after a step. `content` is the gateway's wrapped, untrusted text."""

    model_config = ConfigDict(frozen=True)

    tool: str
    status: ToolStatus
    content: str = ""
    detail: str = ""


class ActionPlanner(Protocol):
    async def next(self, question: str, observations: Sequence[Observation]) -> ActionStep | Finish:
        """Decide the next step from the question and what has been observed so far."""
        ...


class ActionStatus(StrEnum):
    COMPLETED = "completed"  # the planner finished with an answer
    NEEDS_APPROVAL = "needs_approval"  # a write step is waiting for a human
    LIMIT_REACHED = "limit_reached"  # a budget or the repeat guard stopped the run
    FAILED = "failed"  # the planner broke


class ActionLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=6, ge=1, le=20)  # planner decisions
    max_tool_calls: int = Field(default=4, ge=1, le=20)
    max_refusals: int = Field(default=2, ge=1, le=10)  # denied/invalid/unknown results


class PendingAction(BaseModel):
    """Everything needed to resume after a human approves (plain data, checkpoint-safe)."""

    model_config = ConfigDict(frozen=True)

    step: ActionStep
    request: ApprovalRequest
    observations: tuple[Observation, ...] = ()
    tool_calls_used: int = 0
    steps_used: int = 0


class ActionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ActionStatus
    answer: str = ""
    observations: tuple[Observation, ...] = ()
    pending: PendingAction | None = None
    reason: str = ""
    steps_used: int = 0
    tool_calls_used: int = 0


_REFUSALS = {ToolStatus.DENIED, ToolStatus.INVALID_ARGUMENTS, ToolStatus.UNKNOWN_TOOL}


def _observe(result: ToolResult) -> Observation:
    return Observation(
        tool=result.tool, status=result.status, content=result.content, detail=result.detail
    )


async def run_action_loop(
    invoker: ToolInvoker,
    context: SecurityContext,
    planner: ActionPlanner,
    question: str,
    *,
    limits: ActionLimits | None = None,
    resume: PendingAction | None = None,
    grant: str | None = None,
) -> ActionOutcome:
    """Run until the planner finishes or a limit/approval stops it. Never raises for tool or
    planner failures; those become an outcome the caller can show."""
    limits = limits or ActionLimits()
    observations: list[Observation] = list(resume.observations) if resume else []
    seen: set[str] = set()
    steps = resume.steps_used if resume else 0
    calls = resume.tool_calls_used if resume else 0
    refusals = sum(1 for o in observations if o.status in _REFUSALS)
    queued: tuple[ActionStep, str | None] | None = (
        (resume.step, grant) if resume is not None else None
    )

    def outcome(status: ActionStatus, **extra: Any) -> ActionOutcome:
        return ActionOutcome(
            status=status,
            observations=tuple(observations),
            steps_used=steps,
            tool_calls_used=calls,
            **extra,
        )

    while True:
        if queued is not None:
            step, approval = queued
            queued = None
        else:
            if steps >= limits.max_steps:
                return outcome(ActionStatus.LIMIT_REACHED, reason="max_steps")
            steps += 1
            try:
                decision = await planner.next(question, tuple(observations))
            except Exception:  # noqa: BLE001 - a planner bug must not crash the request
                return outcome(ActionStatus.FAILED, reason="planner_error")
            if isinstance(decision, Finish):
                return outcome(ActionStatus.COMPLETED, answer=decision.answer)
            step, approval = decision, None
        if calls >= limits.max_tool_calls:
            return outcome(ActionStatus.LIMIT_REACHED, reason="max_tool_calls")
        fingerprint = step.fingerprint()
        if approval is None and fingerprint in seen:
            return outcome(ActionStatus.LIMIT_REACHED, reason="repeated_call")
        seen.add(fingerprint)
        calls += 1
        result = await invoker.call(context, step.tool, step.arguments, approval=approval)
        if result.status is ToolStatus.APPROVAL_REQUIRED and result.approval_request is not None:
            pending = PendingAction(
                step=step,
                request=result.approval_request,
                observations=tuple(observations),
                tool_calls_used=calls - 1,  # the call did not run; it is retried on resume
                steps_used=steps,
            )
            return outcome(ActionStatus.NEEDS_APPROVAL, pending=pending, reason="write_tool")
        observations.append(_observe(result))
        if result.status in _REFUSALS:
            refusals += 1
            if refusals >= limits.max_refusals:
                return outcome(ActionStatus.LIMIT_REACHED, reason="too_many_refusals")


async def run_action_node(
    state: Mapping[str, Any],
    *,
    invoker: ToolInvoker,
    planner: ActionPlanner,
    limits: ActionLimits | None = None,
) -> dict[str, Any]:
    """Graph node body. Reads `security`, `question` and, when resuming after an approval,
    `action_resume` (a PendingAction dict) and `action_grant` (the signed grant). Returns a
    partial state update with plain data only; it never touches `security` or `question`."""
    context = state["security"]
    resume_raw = state.get("action_resume")
    resume = PendingAction.model_validate(resume_raw) if resume_raw else None
    outcome = await run_action_loop(
        invoker,
        context,
        planner,
        state["question"],
        limits=limits,
        resume=resume,
        grant=state.get("action_grant"),
    )
    update: dict[str, Any] = {
        "action_status": outcome.status.value,
        "action_outcome": outcome.model_dump(mode="json"),
        "action_resume": None,
        "action_grant": None,
    }
    if outcome.status is ActionStatus.COMPLETED:
        update["answer"] = outcome.answer
    if outcome.pending is not None:
        update["action_resume"] = outcome.pending.model_dump(mode="json")
    return update
