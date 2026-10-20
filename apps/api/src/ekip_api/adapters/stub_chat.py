"""Placeholder answer streamer (Phase 2). NO language model: it only restates what retrieval found,
with numbered citations built by the same ContextAssembler the real agent will use (Phase 6).
Replaced by the LangGraph agent behind the same ChatStreamer port."""

import asyncio
from collections.abc import AsyncIterator

from ekip_contracts.chat import ChatEvent
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import Retriever, SearchScope
from ekip_retrieval import ContextAssembler, ContextConfig, InvalidQueryError

from ekip_api.adapters.events import citations_event


class StubChatStreamer:
    def __init__(self, retriever: Retriever, *, token_delay_s: float = 0.0) -> None:
        self._retriever = retriever
        self._assembler = ContextAssembler(ContextConfig(budget_tokens=1500, max_per_document=2))
        self._delay = token_delay_s  # a "typing" pause; lets us observe real streaming end to end

    async def stream(
        self, *, message: str, context: SecurityContext, conversation_id: str | None = None
    ) -> AsyncIterator[ChatEvent]:
        scope = SearchScope.from_context(context)
        try:
            found = await self._retriever.search(message, scope=scope, limit=5)
        except InvalidQueryError:
            found = []
        bundle = self._assembler.assemble(found)
        if bundle.blocks:
            body = "Relevant passages: " + " | ".join(
                f"{b.text} [{b.citation.number}]" for b in bundle.blocks[:3]
            )
        else:
            body = "No relevant documents were found for your question."
        for word in ("[stub answer, no model yet]", *body.split()):
            yield ChatEvent("token", {"text": word + " "})
            if self._delay:
                await asyncio.sleep(self._delay)
        yield citations_event(bundle.citations)
