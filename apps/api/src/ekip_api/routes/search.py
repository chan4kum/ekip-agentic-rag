from typing import Annotated

from ekip_contracts.authorization import Action
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import SearchScope
from fastapi import APIRouter, Depends, HTTPException, Query

from ekip_api.dependencies import get_retrieval_service, limit_requests, require_permission
from ekip_api.schemas import SearchResponse
from ekip_api.services.retrieval import (
    MAX_LIMIT,
    MAX_QUERY_CHARS,
    InvalidQueryError,
    RetrievalService,
)

router = APIRouter(tags=["search"])


@router.get("/api/v1/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY_CHARS)],
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    ctx: Annotated[SecurityContext, Depends(require_permission(Action.DOCUMENTS_SEARCH))],
    _rate_limited: Annotated[None, Depends(limit_requests("search"))],
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 5,
) -> SearchResponse:
    try:
        results = await service.search(q, scope=SearchScope.from_context(ctx), limit=limit)
    except InvalidQueryError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return SearchResponse(query=q.strip(), results=results)
