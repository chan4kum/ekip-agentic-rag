"""Graph nodes. Each is built by a factory that receives its dependencies (constructor injection),
takes the state, and returns a PARTIAL update. A node never mutates the state it was given."""

from collections.abc import Awaitable, Callable
from typing import Any

from ekip_contracts.conversation import ConversationKey, ConversationStore, Turn
from ekip_contracts.embedding import EmbeddingError
from ekip_contracts.llm import ChatModel, LLMError
from ekip_contracts.retrieval import Retriever, SearchScope
from ekip_contracts.search_index import SearchIndexError
from ekip_observability.logging import get_logger
from ekip_retrieval import ContextAssembler, InvalidQueryError, check_citations

from ekip_agents.prompts import build_answer_messages
from ekip_agents.rewrite import QueryRewriter
from ekip_agents.router import QueryRouter
from ekip_agents.state import AgentState, RunStatus, StateViolation, check_update

log = get_logger("ekip.agents")

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]


def _steps(state: AgentState) -> int:
    return state.get("steps", 0) + 1


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
            try:
                reply = await model.complete(
                    build_answer_messages(state["question"], context, state.get("history", [])),
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
        check_update(update)
        return update

    return rewrite
