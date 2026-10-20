"""Serve /api/v1/chat with the LangGraph agent behind the ChatStreamer port.

The graph runs to completion first (it has to verify the citations before anything is shown), and
the verified answer is then streamed to the client in small chunks. Token-by-token streaming from
the model would show unverified text, so it is deliberately not done here.
"""

import re
import uuid
from collections.abc import AsyncIterator

from ekip_agents import FAILURE_MESSAGE, Graph, Grounded, RunStatus, run_agent
from ekip_contracts.chat import ChatEvent
from ekip_contracts.identity import SecurityContext

from ekip_api.adapters.events import citations_event

_CHUNK = re.compile(r"\S+\s*")  # a word plus the whitespace after it; joining chunks restores text


class AgentChatStreamer:
    def __init__(
        self, graph: Graph, *, token_delay_s: float = 0.0, checkpointed: bool = False
    ) -> None:
        self._graph = graph
        self._delay = token_delay_s
        self._checkpointed = checkpointed  # the graph saves state, so each run needs its own thread

    async def stream(
        self, *, message: str, context: SecurityContext, conversation_id: str | None = None
    ) -> AsyncIterator[ChatEvent]:
        thread_key = uuid.uuid4().hex if self._checkpointed else None
        final = await run_agent(
            self._graph, message, context, conversation_id, thread_key=thread_key
        )
        status = final.get("status", RunStatus.FAILED)
        text = final.get("answer") or FAILURE_MESSAGE
        for chunk in _CHUNK.findall(text):
            yield ChatEvent("token", {"text": chunk})
        bundle = final.get("context")
        report = final.get("citations")
        if status is RunStatus.ANSWERED and bundle is not None and report is not None:
            cited = set(report.cited)  # only sources the answer actually cites
            yield citations_event(c for c in bundle.citations if c.number in cited)
        else:
            yield citations_event([])
        grounded = final.get("grounded", Grounded.UNCHECKED)
        yield ChatEvent(
            "status",
            {
                "status": status.value,
                "reason": final.get("reason", ""),
                # which budget cut the run short ("" = none): the answer may rest on less evidence
                "truncated": final.get("truncated", ""),
                # did a judge confirm the sources support the claims? "unchecked" = it did not run
                "grounded": grounded.value,
            },
        )
