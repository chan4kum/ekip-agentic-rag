from fastapi import APIRouter, Request, Response

from ekip_api.schemas import ReadyResponse

router = APIRouter(tags=["ops"])


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request, response: Response) -> ReadyResponse:
    checks: dict[str, bool] = await request.app.state.readiness.run()
    all_ok = all(checks.values())
    if not all_ok:
        response.status_code = 503
    return ReadyResponse(
        status="ready" if all_ok else "not_ready",
        checks={name: "ok" if ok else "fail" for name, ok in checks.items()},
    )
