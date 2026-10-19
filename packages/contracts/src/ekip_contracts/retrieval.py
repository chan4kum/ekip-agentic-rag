from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    text: str
    score: float = Field(ge=0)


class Retriever(Protocol):
    """Port: anything that can search chunks on behalf of ONE tenant."""

    async def search(
        self, query: str, *, tenant_id: str, limit: int = 5
    ) -> list[RetrievedChunk]: ...
