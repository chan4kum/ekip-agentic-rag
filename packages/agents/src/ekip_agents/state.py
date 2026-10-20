"""The agent's working memory for ONE question.

Every graph node reads this state and returns a *partial update* to it. Nodes never talk to each
other directly, so a well-typed state is what makes the graph testable and its failures visible.

Rules this module enforces:
- `security` (who is asking) and `question` are set once, at the start, and no node may replace
  them. Retrieval scope is always derived from `security`, never from model output (invariant I-2).
- Everything in the state is plain data (no clients, no connections), because it will be
  checkpointed to Postgres in a later step.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, TypedDict

from ekip_contracts.context import ContextBundle
from ekip_contracts.conversation import CONVERSATION_ID, Turn
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import RetrievedChunk
from ekip_retrieval import CitationReport

MAX_QUESTION_CHARS = 4000  # same limit as the chat API request


class Route(StrEnum):
    KNOWLEDGE = "knowledge"  # answer from the company's documents
    OUT_OF_SCOPE = "out_of_scope"  # greeting, chit-chat, or something we do not answer


class RunStatus(StrEnum):
    RUNNING = "running"
    ANSWERED = "answered"  # an answer with valid citations
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"  # nothing readable answers the question
    REFUSED = "refused"  # out of scope or blocked by a policy check
    FAILED = "failed"  # a dependency broke; no answer was produced


class AgentState(TypedDict, total=False):
    """`total=False`: a node only has to supply the keys it changes."""

    # set once, immutable
    question: str
    security: SecurityContext
    conversation_id: str  # optional; names the chat thread, validated in new_state
    # produced by nodes
    route: Route
    history: list[Turn]  # earlier ANSWERED turns of this thread, oldest first
    search_query: str  # what retrieval searches for; defaults to `question`
    hits: list[RetrievedChunk]
    context: ContextBundle
    answer: str
    citations: CitationReport
    status: RunStatus
    reason: str  # short code explaining a non-ANSWERED ending (for logs and metrics)
    steps: int  # nodes executed so far; the graph stops at its budget (Phase 7)


PROTECTED_KEYS: frozenset[str] = frozenset({"question", "security", "conversation_id"})
STATE_KEYS: frozenset[str] = frozenset(AgentState.__annotations__)


class StateViolation(Exception):
    """A node tried to change something it must not (or invented a key)."""


def new_state(
    question: str, security: SecurityContext, conversation_id: str | None = None
) -> AgentState:
    """The only way a run begins. `conversation_id` only NAMES a thread; whose thread it is comes
    from `security` (the token), so a guessed id never reaches another user's messages."""
    cleaned = question.strip()
    if not cleaned:
        raise StateViolation("question must not be empty")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise StateViolation("question too long")
    state = AgentState(
        question=cleaned, security=security, status=RunStatus.RUNNING, steps=0, hits=[]
    )
    if conversation_id is not None:
        if not CONVERSATION_ID.fullmatch(conversation_id):
            raise StateViolation("invalid conversation_id")
        state["conversation_id"] = conversation_id
    return state


def check_update(update: Mapping[str, Any]) -> None:
    """Reject a node's return value that touches protected or unknown keys."""
    protected = PROTECTED_KEYS & update.keys()
    if protected:
        raise StateViolation(f"nodes may not change {sorted(protected)}")
    unknown = update.keys() - STATE_KEYS
    if unknown:
        raise StateViolation(f"unknown state keys {sorted(unknown)}")


def apply_update(state: AgentState, update: Mapping[str, Any]) -> AgentState:
    """Merge a node's partial update into the state (what the graph runtime does for us later)."""
    check_update(update)
    return AgentState(**{**state, **update})  # type: ignore[typeddict-item]
