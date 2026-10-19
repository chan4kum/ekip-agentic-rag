from importlib.metadata import version

from fastapi import APIRouter, Request

from ekip_api.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="ekip-api",
        version=version("ekip-api"),
        environment=request.app.state.settings.app_env,
    )
