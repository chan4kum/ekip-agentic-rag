from typing import Annotated, Literal

from ekip_contracts.retrieval import RetrievedChunk
from pydantic import BaseModel, ConfigDict, StringConstraints


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    service: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ready", "not_ready"]
    checks: dict[str, Literal["ok", "fail"]]


class SearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    results: list[RetrievedChunk]


class ChatRequest(BaseModel):
    # extra="forbid": a caller who sends tenant_id (or anything unexpected) gets a 422, not silence.
    model_config = ConfigDict(frozen=True, extra="forbid")

    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    conversation_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{8,64}$")] | None = (
        None
    )
