from typing import Literal

from ekip_contracts.retrieval import RetrievedChunk
from pydantic import BaseModel, ConfigDict


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
