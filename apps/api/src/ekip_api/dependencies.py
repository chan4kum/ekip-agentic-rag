from fastapi import HTTPException, Request

from ekip_api.services.retrieval import RetrievalService


def get_retrieval_service(request: Request) -> RetrievalService:
    return RetrievalService(request.app.state.retriever)


def get_tenant_id(request: Request) -> str:
    """TEMPORARY until step 2.3 (validated JWT -> SecurityContext).

    The tenant is decided by the server, never by the request (I-4).
    Fails closed outside local/test (I-1).
    """
    if request.app.state.settings.app_env in {"local", "test"}:
        return "tenant-dev"
    raise HTTPException(status_code=401, detail="authentication required")
