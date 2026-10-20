"""Graph nodes. Each is built by a factory that receives its dependencies (constructor injection),
takes the state, and returns a PARTIAL update. A node never mutates the state it was given."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ekip_contracts.context import ContextBundle
from ekip_contracts.conversation import ConversationKey, ConversationStore, Turn
from ekip_contracts.embedding import EmbeddingError
from ekip_contracts.llm import ChatModel, LLMError
from ekip_contracts.retrieval import RetrievedChunk, Retriever, SearchScope
from ekip_contracts.search_index import SearchIndexError
from ekip_observability.logging import get_logger
from ekip_retrieval import ContextAssembler, InvalidQueryError, check_citations

from ekip_agents.budget import Budget, BudgetExceeded, BudgetTracker, Resource
from ekip_agents.complexity import Complexity, ComplexityRouter
from ekip_agents.evidence import merge_hits
from ekip_agents.groundedness import UNCHECKED, GroundednessChecker, GroundednessResult
from ekip_agents.plan import single_task_plan
from ekip_agents.planner import QueryPlanner
from ekip_agents.prompts import build_answer_messages, build_repair_messages
from ekip_agents.rewrite import QueryRewriter
from ekip_agents.router import QueryRouter
from ekip_agents.state import AgentState, Grounded, RunStatus, StateViolation, check_update

log = get_logger("ekip.agents")

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _steps(state: AgentState) -> int:
    return state.get("steps", 0) + 1


def _calls(state: AgentState, spent: int = 1) -> int:
    return state.get("model_calls", 0) + spent


def make_classify_node(router: QueryRouter) -> Node:
    async def classify(state: AgentState) -> dict[str, Any]:
        if "question" not in state:
            raise StateViolation("classify needs a question in the state")
        update: dict[str, Any] = {
            "route": await router.classify(state["question"]),
            "steps": _steps(state),
        }
        check_update(update)  # a node that writes forbidden keys fails here, in every test
        return update

    return classify


def make_retrieve_node(
    retriever: Retriever, assembler: ContextAssembler, *, limit: int = 8
) -> Node:
    """Search the documents AS THE CALLER and turn the hits into a citable context.

    The scope (tenant, user, groups) is built from `state["security"]` only - the verified token.
    Nothing the question says, and nothing a model produced, can widen it (invariant I-2).
    """

    async def retrieve(state: AgentState) -> dict[str, Any]:
        if "question" not in state or "security" not in state:
            raise StateViolation("retrieve needs a question and a security context")
        scope = SearchScope.from_context(state["security"])
        update: dict[str, Any] = {"steps": _steps(state)}
        try:
            query = state.get("search_query") or state["question"]
            hits = await retriever.search(query, scope=scope, limit=limit)
        except InvalidQueryError:
            update.update(status=RunStatus.REFUSED, hits=[])
        except (SearchIndexError, EmbeddingError) as exc:
            # reason code only: exception text could echo query or document content
            log.warning("retrieve_failed", error_type=type(exc).__name__)
            update.update(status=RunStatus.FAILED, hits=[])
        else:
            bundle = assembler.assemble(hits)
            update.update(hits=hits, context=bundle)
            if not bundle.blocks:  # nothing readable answers it: say so, do not guess
                update["status"] = RunStatus.INSUFFICIENT_EVIDENCE
        check_update(update)
        return update

    return retrieve


def _sub_questions(state: AgentState) -> list[str]:
    """Sub-questions to show the answer model: only for COMPLEX runs that really were split."""
    plan = state.get("plan")
    if plan is None or state.get("complexity") is not Complexity.COMPLEX:
        return []
    return [t.question for t in plan.subtasks]


def make_answer_node(model: ChatModel, *, max_tokens: int = 800) -> Node:
    """Ask the model for an answer grounded in the retrieved context.

    No context -> no model call: nothing to ground on means "insufficient evidence", and it saves
    cost. The reply is only DRAFT text here; the next node verifies its citations.
    """

    async def answer(state: AgentState) -> dict[str, Any]:
        if "question" not in state:
            raise StateViolation("answer needs a question in the state")
        update: dict[str, Any] = {"steps": _steps(state)}
        context = state.get("context")
        if context is None or not context.blocks:
            update["status"] = RunStatus.INSUFFICIENT_EVIDENCE
        else:
            update["model_calls"] = _calls(state)
            try:
                reply = await model.complete(
                    build_answer_messages(
                        state["question"],
                        context,
                        state.get("history", []),
                        _sub_questions(state),
                    ),
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            except LLMError as exc:
                log.warning("answer_failed", error_type=type(exc).__name__, model=model.model_name)
                update["status"] = RunStatus.FAILED
            else:
                update["answer"] = reply.strip()
        check_update(update)
        return update

    return answer


def make_verify_node() -> Node:
    """Accept the draft answer only if it is grounded in numbered sources that really exist.

    What this proves: every `[n]` points at a source we gave the model, and the answer cites at
    least one. What it does NOT prove: that the cited source actually supports the claim. That
    needs a groundedness check (the Phase 7 verifier and the evaluation suite).

    A rejected draft is erased from the state so no later node can leak it.
    """

    async def verify(state: AgentState) -> dict[str, Any]:
        if "answer" not in state or "context" not in state:
            raise StateViolation("verify needs an answer and a context in the state")
        report = check_citations(state["answer"], state["context"])
        update: dict[str, Any] = {"steps": _steps(state), "citations": report}
        if not report.ok:  # the model invented source numbers
            update.update(status=RunStatus.FAILED, reason="invalid_citations", answer="")
        elif not report.has_citations:  # nothing to check the claims against
            update.update(status=RunStatus.INSUFFICIENT_EVIDENCE, reason="no_citations", answer="")
        else:
            update["status"] = RunStatus.ANSWERED
        check_update(update)
        return update

    return verify


def make_load_history_node(store: ConversationStore, *, limit: int = 6) -> Node:
    """Load this thread's earlier ANSWERED turns. Memory is best-effort: if the store is down or
    the key is invalid the run continues with no history (never with someone else's)."""

    async def load_history(state: AgentState) -> dict[str, Any]:
        update: dict[str, Any] = {"steps": _steps(state), "history": []}
        conversation_id = state.get("conversation_id")
        if conversation_id is not None and "security" in state:
            try:
                key = ConversationKey.from_context(state["security"], conversation_id)
                update["history"] = await store.recent(key, limit)
            except Exception as exc:  # best effort by design; type only, text may hold content
                log.warning("history_load_failed", error_type=type(exc).__name__)
        check_update(update)
        return update

    return load_history


def make_save_turn_node(store: ConversationStore) -> Node:
    """Remember the turn - but ONLY if it ended ANSWERED. Refusals, failures and 'no evidence'
    are never stored, so memory holds nothing but verified, cited answers."""

    async def save_turn(state: AgentState) -> dict[str, Any]:
        update: dict[str, Any] = {"steps": _steps(state)}
        conversation_id = state.get("conversation_id")
        if (
            state.get("status") is RunStatus.ANSWERED
            and conversation_id is not None
            and "security" in state
            and "question" in state
            and "answer" in state
        ):
            try:
                key = ConversationKey.from_context(state["security"], conversation_id)
                await store.append(key, Turn(state["question"], state["answer"]))
            except (
                Exception
            ) as exc:  # the user already has the answer; do not turn it into a failure
                log.warning("turn_save_failed", error_type=type(exc).__name__)
        check_update(update)
        return update

    return save_turn


def make_rewrite_node(rewriter: QueryRewriter) -> Node:
    """Make a follow-up searchable on its own. Only `search_query` changes; the question the user
    asked, and the answer prompt, still use the original wording."""

    async def rewrite(state: AgentState) -> dict[str, Any]:
        if "question" not in state:
            raise StateViolation("rewrite needs a question in the state")
        history = state.get("history", [])
        query = await rewriter.rewrite(state["question"], history) if history else state["question"]
        update: dict[str, Any] = {"steps": _steps(state), "search_query": query}
        if history:  # the rewriter calls the model only when there is history to resolve
            update["model_calls"] = _calls(state)
        check_update(update)
        return update

    return rewrite


def make_plan_node(router: ComplexityRouter, planner: QueryPlanner) -> Node:
    """Decide how many searches the question needs.

    SIMPLE questions get a one-task plan with no model call. Only COMPLEX ones pay for planning.
    The plan starts from `search_query` (already standalone after rewriting) when it exists.
    """

    async def plan(state: AgentState) -> dict[str, Any]:
        if "question" not in state:
            raise StateViolation("plan needs a question in the state")
        query = state.get("search_query") or state["question"]
        complexity = await router.assess(query)
        the_plan = (
            await planner.plan(query)
            if complexity is Complexity.COMPLEX
            else single_task_plan(query)
        )
        update: dict[str, Any] = {
            "complexity": complexity,
            "plan": the_plan,
            "steps": _steps(state),
        }
        if complexity is Complexity.COMPLEX:  # only complex questions pay for a planner call
            update["model_calls"] = _calls(state)
        check_update(update)
        return update

    return plan


async def _search_task(
    retriever: Retriever, question: str, scope: SearchScope, limit: int
) -> tuple[str, list[RetrievedChunk]]:
    """One sub-question. Returns ('ok'|'invalid'|'failed', hits). Never raises for search errors."""
    try:
        return "ok", await retriever.search(question, scope=scope, limit=limit)
    except InvalidQueryError:
        return "invalid", []
    except (SearchIndexError, EmbeddingError) as exc:
        log.warning("retrieve_failed", error_type=type(exc).__name__)  # type only, no text
        return "failed", []


def make_execute_node(
    retriever: Retriever,
    assembler: ContextAssembler,
    budget: Budget,
    *,
    limit: int = 6,
    clock: Callable[[], float] = time.monotonic,
) -> Node:
    """Run the plan: searches within a wave run in parallel, waves run in dependency order.

    Every search is AS THE CALLER (scope from the verified token only) and costs one retrieval
    from the budget. A search that does not fit the budget is skipped and the run continues with
    the evidence it has (reported in `truncated`). One failing sub-question does not sink the
    others; only if nothing could be searched does the run end FAILED / REFUSED.
    """

    async def execute(state: AgentState) -> dict[str, Any]:
        if "question" not in state or "security" not in state or "plan" not in state:
            raise StateViolation("execute needs a question, a security context and a plan")
        scope = SearchScope.from_context(state["security"])
        tracker = BudgetTracker(budget, clock=clock)
        per_task: dict[str, list[RetrievedChunk]] = {}
        outcomes: dict[str, int] = {"ok": 0, "invalid": 0, "failed": 0}
        truncated = ""
        for wave in state["plan"].waves():
            runnable = []
            for task in wave:
                try:
                    tracker.spend(Resource.RETRIEVALS)
                except BudgetExceeded as exc:
                    truncated = exc.resource
                    continue
                runnable.append(task)
            results = await asyncio.gather(
                *(_search_task(retriever, t.question, scope, limit) for t in runnable)
            )
            for task, (outcome, hits) in zip(runnable, results, strict=True):
                outcomes[outcome] += 1
                if outcome == "ok":
                    per_task[task.id] = hits
            if truncated:
                break  # do not start later waves once the budget is gone
        update: dict[str, Any] = {"steps": _steps(state)}
        if truncated:
            update["truncated"] = truncated
        merged = merge_hits(list(per_task.values()))
        update["hits"] = merged
        update["task_hits"] = {k: [h.chunk_id for h in v] for k, v in per_task.items()}
        if outcomes["ok"] == 0:  # nothing was searched successfully
            if outcomes["failed"] or truncated:
                update.update(status=RunStatus.FAILED, reason="search_unavailable")
            else:
                update.update(status=RunStatus.REFUSED, reason="invalid_query")
        else:
            bundle = assembler.assemble(merged)
            update["context"] = bundle
            if not bundle.blocks:  # nothing readable answers it: say so, do not guess
                update["status"] = RunStatus.INSUFFICIENT_EVIDENCE
        check_update(update)
        return update

    return execute


async def _safe_check(
    checker: GroundednessChecker, answer: str, context: ContextBundle
) -> GroundednessResult:
    try:
        return await checker.check(answer, context)
    except Exception as exc:  # a broken judge must never break the answer path; type only
        log.warning("ground_crashed", error_type=type(exc).__name__)
        return UNCHECKED


def make_ground_node(
    checker: GroundednessChecker, model: ChatModel, budget: Budget, *, max_tokens: int = 800
) -> Node:
    """Check that the cited sources SUPPORT the answer, after the citation check passed.

    - grounded: the answer stays
    - unchecked (judge down / unusable / out of model-call budget): the answer stays on its
      citation check alone and the state says so, so the client can show it
    - ungrounded: ONE repair attempt (needs two spare model calls: rewrite + re-check); if the
      repaired answer is not confirmed grounded, the run ends INSUFFICIENT_EVIDENCE and the
      unsupported draft is erased
    """

    def reject(update: dict[str, Any], calls: int) -> None:
        update.update(
            status=RunStatus.INSUFFICIENT_EVIDENCE,
            reason="ungrounded",
            answer="",
            grounded=Grounded.UNGROUNDED,
            model_calls=calls,
        )

    async def ground(state: AgentState) -> dict[str, Any]:
        update: dict[str, Any] = {}
        if (
            state.get("status") is not RunStatus.ANSWERED
            or "answer" not in state
            or "context" not in state
            or "question" not in state
        ):
            check_update(update)
            return update  # nothing to check (an earlier node already ended the run)
        update["steps"] = _steps(state)
        answer, context, calls = state["answer"], state["context"], state.get("model_calls", 0)
        if calls + 1 > budget.max_model_calls:
            update.update(
                grounded=Grounded.UNCHECKED, truncated=state.get("truncated") or "model_calls"
            )
            check_update(update)
            return update
        result = await _safe_check(checker, answer, context)
        calls += 1
        update["model_calls"] = calls
        update["grounded"] = result.status
        if result.status is Grounded.UNGROUNDED:
            if calls + 2 > budget.max_model_calls:  # cannot afford repair + re-check
                reject(update, calls)
                update["truncated"] = state.get("truncated") or "model_calls"
            else:
                try:
                    draft = await model.complete(
                        build_repair_messages(
                            state["question"],
                            context,
                            answer,
                            result.unsupported,
                            state.get("history", []),
                            _sub_questions(state),
                        ),
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                except LLMError as exc:
                    log.warning("repair_failed", error_type=type(exc).__name__)
                    reject(update, calls + 1)
                else:
                    repaired = draft.strip()
                    report = check_citations(repaired, context)
                    calls += 1
                    if not report.ok or not report.has_citations:
                        reject(update, calls)
                    else:
                        again = await _safe_check(checker, repaired, context)
                        calls += 1
                        if again.status is Grounded.GROUNDED:
                            update.update(
                                answer=repaired,
                                citations=report,
                                grounded=Grounded.GROUNDED,
                                model_calls=calls,
                            )
                        else:
                            reject(update, calls)
        check_update(update)
        return update

    return ground
