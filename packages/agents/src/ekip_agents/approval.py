"""Human approval as a graph step, built on LangGraph's `interrupt`.

The gate pauses the run and hands a small, fixed request (action name + summary) to a human. The
run is saved by the checkpointer and survives restarts; a later `resume_run` call with the same
thread continues it. Rules:
- anything other than the exact word "approve" counts as a rejection (fail closed)
- the thread id contains tenant + user, and a resume must come from the same verified identity;
  another user can neither see nor resume someone else's pending run
- a rejected run ends REFUSED and the withheld answer is erased
- nothing before `interrupt` has side effects: LangGraph re-runs the node from its start on resume
"""

from dataclasses import dataclass
from typing import Any

from ekip_contracts.identity import SecurityContext
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from ekip_agents.nodes import Node
from ekip_agents.state import (
    AgentState,
    Approval,
    RunStatus,
    StateViolation,
    check_update,
    new_state,
)

APPROVE = "approve"
REJECT = "reject"
NOT_APPROVED_MESSAGE = "This request was not approved, so it was not completed."


class ApprovalError(Exception):
    """Resume was attempted for a run that is missing, not waiting, or not the caller's."""


@dataclass(frozen=True)
class ApprovalGate:
    """What a human is asked. Fixed text only - never model output or document content."""

    action: str
    summary: str


@dataclass(frozen=True)
class ApprovalRequest:
    action: str
    summary: str


def make_approval_node(gate: ApprovalGate) -> Node:
    async def approval(state: AgentState) -> dict[str, Any]:
        update: dict[str, Any] = {}
        if state.get("status") is not RunStatus.ANSWERED:
            check_update(update)
            return update  # an earlier node already ended the run; nothing to approve
        decision = interrupt({"action": gate.action, "summary": gate.summary})
        update["steps"] = state.get("steps", 0) + 1
        if decision == APPROVE:
            update["approval"] = Approval.APPROVED
        else:
            update.update(
                approval=Approval.REJECTED,
                status=RunStatus.REFUSED,
                reason="approval_rejected",
                answer=NOT_APPROVED_MESSAGE,
            )
        check_update(update)
        return update

    return approval


def thread_config(security: SecurityContext, thread_key: str) -> RunnableConfig:
    """LangGraph thread id = tenant + user + caller-chosen key, so threads never cross users."""
    if not thread_key or len(thread_key) > 64 or ":" in thread_key:
        raise StateViolation("invalid thread key")
    return {"configurable": {"thread_id": f"{security.tenant_id}:{security.subject}:{thread_key}"}}


Graph = CompiledStateGraph[AgentState, Any, AgentState, AgentState]


async def _pending(graph: Graph, config: RunnableConfig) -> ApprovalRequest | None:
    snapshot = await graph.aget_state(config)
    for task in snapshot.tasks:
        for pending in task.interrupts:
            value = pending.value
            if isinstance(value, dict):
                return ApprovalRequest(str(value.get("action", "")), str(value.get("summary", "")))
    return None


async def start_run(
    graph: Graph, question: str, security: SecurityContext, thread_key: str
) -> tuple[AgentState, ApprovalRequest | None]:
    """Run until the end or until a human decision is needed."""
    config = thread_config(security, thread_key)
    result = await graph.ainvoke(new_state(question, security), config)
    return _values(result), await _pending(graph, config)


async def pending_approval(
    graph: Graph, security: SecurityContext, thread_key: str
) -> ApprovalRequest | None:
    return await _pending(graph, thread_config(security, thread_key))


async def resume_run(
    graph: Graph, security: SecurityContext, thread_key: str, *, approve: bool
) -> AgentState:
    """Continue a paused run with a human decision. Raises ApprovalError if nothing of the
    caller's is waiting."""
    config = thread_config(security, thread_key)
    snapshot = await graph.aget_state(config)
    if not snapshot.values or snapshot.values.get("security") != security:
        raise ApprovalError("no pending run for this caller")
    if await _pending(graph, config) is None:
        raise ApprovalError("run is not waiting for approval")
    result = await graph.ainvoke(Command(resume=APPROVE if approve else REJECT), config)
    return _values(result)


def _values(result: Any) -> AgentState:
    state = {k: v for k, v in dict(result).items() if not k.startswith("__")}
    return state  # type: ignore[return-value]
