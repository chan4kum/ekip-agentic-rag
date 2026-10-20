"""Conversation memory: the question/answer turns of one chat thread.

Security by construction: a thread is addressed by (tenant, USER, conversation id), all three. A
caller can only ever name the third; the first two come from the verified token. So guessing or
stealing someone else's conversation id yields an empty history, never their messages.

What is stored: only turns that ended ANSWERED (question + cited answer). Not retrieved passages
and not graph state, so memory cannot outlive a permission revocation by copying document text.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ekip_contracts.documents import TENANT_ID, ContractError
from ekip_contracts.identity import SecurityContext

CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")  # same shape the chat API accepts
MAX_TURN_CHARS = 8000
MAX_TURNS_KEPT = 50  # per conversation; older turns are dropped on append
SUBJECT = re.compile(r"^[\x21-\x7e]{1,256}$")  # printable, no spaces


@dataclass(frozen=True)
class ConversationKey:
    tenant_id: str
    user: str  # the token's subject, never client supplied
    conversation_id: str

    def __post_init__(self) -> None:
        if not TENANT_ID.fullmatch(self.tenant_id):
            raise ContractError("invalid tenant_id")
        if not SUBJECT.fullmatch(self.user):
            raise ContractError("invalid user")
        if not CONVERSATION_ID.fullmatch(self.conversation_id):
            raise ContractError("invalid conversation_id")

    @classmethod
    def from_context(cls, context: SecurityContext, conversation_id: str) -> "ConversationKey":
        return cls(context.tenant_id, context.subject, conversation_id)


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
    created_at: datetime | None = None  # stamped by the store

    def __post_init__(self) -> None:
        for name, value in (("question", self.question), ("answer", self.answer)):
            if not value.strip():
                raise ContractError(f"{name} must not be empty")
            if len(value) > MAX_TURN_CHARS:
                raise ContractError(f"{name} too long")


class ConversationStore(Protocol):
    async def append(self, key: ConversationKey, turn: Turn) -> None:
        """Add a turn (newest). Keeps only the latest MAX_TURNS_KEPT per conversation."""
        ...

    async def recent(self, key: ConversationKey, limit: int = 6) -> list[Turn]:
        """The latest `limit` turns, OLDEST first. Unknown conversation -> empty list."""
        ...

    async def delete(self, key: ConversationKey) -> int:
        """Remove the whole conversation; returns how many turns were deleted."""
        ...
