"""Placeholder answer streamer (Phase 2). NO language model: it only restates what retrieval found.
Replaced in Phase 6 by the LangGraph agent behind the same ChatStreamer port."""

import asyncio
from collections.abc import AsyncIterator

from ekip_contracts.chat import ChatEvent
from ekip_contracts.retrieval import Retriever


class StubChatStreamer:
    def __init__(self, retriever: Retriever, *, token_delay_s: float = 0.0) -> None:
        self._retriever = retriever
        self._delay = token_delay_s  # a "typing" pause; lets us observe real streaming end to end

    async def stream(self, *, message: str, tenant_id: str) -> AsyncIterator[ChatEvent]:
        found = await self._retriever.search(message, tenant_id=tenant_id, limit=3)
        if found:
            body = "Relevant passages: " + " | ".join(c.text for c in found)
        else:
            body = "No relevant documents were found for your question."
        for word in ("[stub answer, no model yet]", *body.split()):
            yield ChatEvent("token", {"text": word + " "})
            if self._delay:
                await asyncio.sleep(self._delay)
        yield ChatEvent(
            "citations",
            {"results": [{"chunk_id": c.chunk_id, "document_id": c.document_id} for c in found]},
        )
