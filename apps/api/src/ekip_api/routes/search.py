from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from ekip_api.dependencies import get_retrieval_service, get_tenant_id
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
    tenant_id: Annotated[str, Depends(get_tenant_id)],
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 5,
) -> SearchResponse:
    try:
        results = await service.search(q, tenant_id=tenant_id, limit=limit)
    except InvalidQueryError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return SearchResponse(query=q.strip(), results=results)
