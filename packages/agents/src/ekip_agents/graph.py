"""The single-agent RAG graph:

    classify -> (out of scope) -> refuse ------------------------------+
        |                                                              |
        +-> retrieve -> answer -> verify -> finalize <-----------------+
               |          |                    ^
               +-- no evidence / failure ------+   (any node may end the run early)

Every edge decision is a plain function of the state, so routing is deterministic and testable
without a model. `finalize` guarantees a run never ends RUNNING and that a user-facing message
exists for every non-answer outcome. A step budget stops runaway loops (Phase 7 adds loops).
"""

from collections.abc import Callable
from typing import Any, cast

from ekip_contracts.conversation import ConversationStore
from ekip_contracts.identity import SecurityContext
from ekip_contracts.llm import ChatModel
from ekip_contracts.retrieval import Retriever
from ekip_retrieval import ContextAssembler
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ekip_agents.approval import ApprovalGate, make_approval_node, thread_config
from ekip_agents.budget import Budget
from ekip_agents.complexity import ComplexityRouter, RuleBasedComplexityRouter
from ekip_agents.groundedness import GroundednessChecker
from ekip_agents.nodes import (
    Node,
    make_answer_node,
    make_classify_node,
    make_execute_node,
    make_ground_node,
    make_load_history_node,
    make_plan_node,
    make_retrieve_node,
    make_rewrite_node,
    make_save_turn_node,
    make_verify_node,
)
from ekip_agents.planner import QueryPlanner
from ekip_agents.rewrite import QueryRewriter
from ekip_agents.router import QueryRouter
from ekip_agents.state import AgentState, Route, RunStatus, check_update, new_state

OUT_OF_SCOPE_MESSAGE = "I can help with questions about your company's documents."
NO_EVIDENCE_MESSAGE = "I could not find this in the documents you have access to."
FAILURE_MESSAGE = "I could not complete that request. Please try again."

Graph = CompiledStateGraph[AgentState, Any, AgentState, AgentState]


async def refuse(state: AgentState) -> dict[str, Any]:
    update: dict[str, Any] = {
        "status": RunStatus.REFUSED,
        "reason": "out_of_scope",
        "answer": OUT_OF_SCOPE_MESSAGE,
    }
    check_update(update)
    return update


async def finalize(state: AgentState) -> dict[str, Any]:
    """The last node of every path: no run ends RUNNING, and non-answers get a safe message."""
    status = state.get("status", RunStatus.RUNNING)
    update: dict[str, Any] = {}
    if status is RunStatus.RUNNING:  # fell through without a decision: that is a failure
        update.update(status=RunStatus.FAILED, reason="incomplete", answer=FAILURE_MESSAGE)
    elif status is RunStatus.INSUFFICIENT_EVIDENCE:
        update["answer"] = NO_EVIDENCE_MESSAGE
    elif status is RunStatus.FAILED:
        update["answer"] = FAILURE_MESSAGE
    elif status is RunStatus.REFUSED and not state.get("answer"):
        update["answer"] = OUT_OF_SCOPE_MESSAGE
    check_update(update)
    return update


def _guard(next_node: str, max_steps: int) -> Callable[[AgentState], str]:
    """Go to `next_node` unless the run already ended or exhausted its step budget."""

    def route(state: AgentState) -> str:
        if state.get("status", RunStatus.RUNNING) is not RunStatus.RUNNING:
            return "finalize"
        if state.get("steps", 0) >= max_steps:
            return "budget"
        return next_node

    return route


async def _budget_exceeded(state: AgentState) -> dict[str, Any]:
    update: dict[str, Any] = {"status": RunStatus.FAILED, "reason": "step_budget"}
    check_update(update)
    return update


def build_agent_graph(
    *,
    router: QueryRouter,
    retriever: Retriever,
    model: ChatModel,
    assembler: ContextAssembler | None = None,
    max_steps: int = 12,
    store: ConversationStore | None = None,
    rewriter: QueryRewriter | None = None,
    planner: QueryPlanner | None = None,
    complexity: ComplexityRouter | None = None,
    budget: Budget | None = None,
    grounding: GroundednessChecker | None = None,
    approval: ApprovalGate | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Graph:
    """Assemble the agent. Optional parts switch features on:

    - `store`: load_history first, save_turn last (conversation memory)
    - `rewriter`: follow-ups become standalone search queries
    - `planner`: complex questions are split into several searches (plan -> execute); without it
      the single `retrieve` node is used, exactly as in Phase 6
    - `grounding`: after the citation check, a judge confirms the sources support the claims
      (one repair attempt, then INSUFFICIENT_EVIDENCE); without it nothing changes
    - `approval`: before the answer is released (and remembered) a human must approve; needs a
      `checkpointer` so the paused run can be resumed (see `ekip_agents.approval`)
    - `checkpointer`: saves the run state per thread (in memory or Postgres)
    """
    if approval is not None and checkpointer is None:
        raise ValueError("an approval gate needs a checkpointer to pause and resume")
    assembler = assembler or ContextAssembler()
    nodes: dict[str, Node] = {
        "classify": make_classify_node(router),
        "answer": make_answer_node(model),
        "verify": make_verify_node(),
        "refuse": refuse,
        "budget": _budget_exceeded,
        "finalize": finalize,
    }
    pipeline: list[str] = []  # the nodes between classify and verify, in order
    if rewriter is not None:
        nodes["rewrite"] = make_rewrite_node(rewriter)
        pipeline.append("rewrite")
    if planner is not None:
        nodes["plan"] = make_plan_node(complexity or RuleBasedComplexityRouter(), planner)
        nodes["execute"] = make_execute_node(retriever, assembler, budget or Budget())
        pipeline += ["plan", "execute"]
    else:
        nodes["retrieve"] = make_retrieve_node(retriever, assembler)
        pipeline.append("retrieve")
    pipeline += ["answer", "verify"]
    if grounding is not None:
        nodes["ground"] = make_ground_node(grounding, model, budget or Budget())
    if approval is not None:
        nodes["approve"] = make_approval_node(approval)
    if store is not None:
        nodes["load_history"] = make_load_history_node(store)
        nodes["save_turn"] = make_save_turn_node(store)
    graph = StateGraph(AgentState)
    for name, node in nodes.items():
        # LangGraph's overloads cannot infer our async-callable alias; the node signature is fixed
        graph.add_node(name, node)  # type: ignore[call-overload]
    graph.add_edge(START, "load_history" if store is not None else "classify")
    if store is not None:
        graph.add_edge("load_history", "classify")
    first = pipeline[0]
    graph.add_conditional_edges(
        "classify",
        lambda s: "refuse" if s.get("route") is Route.OUT_OF_SCOPE else _guard(first, max_steps)(s),
        {"refuse": "refuse", first: first, "finalize": "finalize", "budget": "budget"},
    )
    for current, following in zip(pipeline, pipeline[1:], strict=False):
        if current == "verify":
            break
        graph.add_conditional_edges(
            current,
            _guard(following, max_steps),
            {following: following, "finalize": "finalize", "budget": "budget"},
        )
    tail = ["verify"]  # what runs after verify, in order, before finalize
    if grounding is not None:
        tail.append("ground")
    if approval is not None:
        tail.append("approve")
    if store is not None:
        tail.append("save_turn")
    for current, following in zip(tail, [*tail[1:], "finalize"], strict=True):
        graph.add_edge(current, following)
    graph.add_edge("refuse", "finalize")
    graph.add_edge("budget", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


async def run_agent(
    graph: Graph,
    question: str,
    security: SecurityContext,
    conversation_id: str | None = None,
    *,
    thread_key: str | None = None,
) -> AgentState:
    """Run one question to completion. Raises StateViolation for an invalid question.

    A graph compiled with a checkpointer needs a `thread_key` (one per run, never reused: a
    reused thread would carry the previous run's state into the next one).
    """
    state = new_state(question, security, conversation_id)
    if thread_key is None:
        final = await graph.ainvoke(state)
    else:
        final = await graph.ainvoke(state, thread_config(security, thread_key))
    return cast(AgentState, final)  # LangGraph types the result loosely; it is our state
