"""Port for producing a chat answer as a stream of events. Phase 6 plugs the agent in here."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ekip_contracts.identity import SecurityContext


@dataclass(frozen=True)
class ChatEvent:
    """One step of an answer. The API adds the framing events (meta, done, error) itself."""

    event: Literal["token", "citations", "status"]
    data: dict[str, Any]


class ChatStreamer(Protocol):
    def stream(
        self, *, message: str, context: SecurityContext, conversation_id: str | None = None
    ) -> AsyncIterator[ChatEvent]:
        """Yield events for one question. `context` is built from the validated token only (I-4);
        retrieval scope (tenant + user + groups) is derived from it, never from the request.
        `conversation_id` only NAMES a thread; whose thread it is comes from `context`.
        Must be cancellable: the caller cancels it when the client disconnects."""
        ...
