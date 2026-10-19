"""Port for producing a chat answer as a stream of events. Phase 6 plugs the agent in here."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ChatEvent:
    """One step of an answer. The API adds the framing events (meta, done, error) itself."""

    event: Literal["token", "citations"]
    data: dict[str, Any]


class ChatStreamer(Protocol):
    def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
        """Yield events for one question. `tenant_id` comes from the validated token only (I-4).
        Must be cancellable: the caller cancels it when the client disconnects."""
        ...
