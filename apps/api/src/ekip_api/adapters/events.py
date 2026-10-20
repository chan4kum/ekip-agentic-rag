"""Event payload helpers shared by the chat streamers."""

from collections.abc import Iterable

from ekip_contracts.chat import ChatEvent
from ekip_contracts.context import Citation


def citations_event(citations: Iterable[Citation]) -> ChatEvent:
    return ChatEvent(
        "citations",
        {
            "results": [
                {
                    "number": c.number,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "document_version": c.document_version,
                    "title": c.title,
                    "source_uri": c.source_uri,
                    "page_number": c.page_number,
                }
                for c in citations
            ]
        },
    )
