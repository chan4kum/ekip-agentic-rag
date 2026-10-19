"""Uniform error responses: RFC 9457 `application/problem+json`.

Rules: the client gets a stable `type`, a short `title`, a `detail` that is safe to show, and the
`request_id` to quote to support. Never the caller's input, a stack trace, or policy internals.
"""

from typing import Any, cast

from ekip_observability.context import new_request_id
from ekip_observability.logging import get_logger
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = get_logger("ekip.api.errors")

PROBLEM_JSON = "application/problem+json"
REQUEST_ID_SCOPE_KEY = "ekip.request_id"  # set by RequestContextMiddleware

_TYPES = {
    400: "bad-request",
    401: "unauthorized",
    403: "forbidden",
    404: "not-found",
    405: "method-not-allowed",
    409: "conflict",
    413: "payload-too-large",
    415: "unsupported-media-type",
    422: "validation-error",
    429: "rate-limited",
    500: "internal-error",
    503: "service-unavailable",
    504: "timeout",
}
_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def request_id_of(scope: Any) -> str:
    rid = scope.get(REQUEST_ID_SCOPE_KEY)
    return rid if isinstance(rid, str) else new_request_id()


def problem_response(
    status: int,
    detail: str,
    *,
    request_id: str,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
    slug: str | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": f"urn:ekip:problem:{slug or _TYPES.get(status, f'http-{status}')}",
        "title": _TITLES.get(status, "Error"),
        "status": status,
        "detail": detail,
        "request_id": request_id,
        **(extra or {}),
    }
    out = {"X-Request-ID": request_id, **(headers or {})}
    return JSONResponse(body, status_code=status, headers=out, media_type=PROBLEM_JSON)


class ProblemError(StarletteHTTPException):
    """An HTTP error with its own stable problem `type` (instead of the generic one per status)."""

    def __init__(
        self, status_code: int, detail: str, *, slug: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(status_code, detail, headers=headers)
        self.slug = slug


async def _http_exception(request: Request, exc: Exception) -> JSONResponse:
    exc = cast(StarletteHTTPException, exc)  # registered for exactly this type
    detail = exc.detail if isinstance(exc.detail, str) else _TITLES.get(exc.status_code, "Error")
    return problem_response(
        exc.status_code,
        detail,
        request_id=request_id_of(request.scope),
        headers=dict(exc.headers or {}),
        slug=getattr(exc, "slug", None),
    )


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    exc = cast(RequestValidationError, exc)  # registered for exactly this type
    # Keep only WHERE and WHAT was wrong. FastAPI's default also echoes `input`, which would
    # reflect the caller's (possibly sensitive) values straight back.
    errors = [
        {
            "loc": [str(p) for p in e.get("loc", ())],
            "msg": e.get("msg", ""),
            "type": e.get("type", ""),
        }
        for e in exc.errors()
    ]
    return problem_response(
        422,
        "request validation failed",
        request_id=request_id_of(request.scope),
        extra={"errors": errors},
    )


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    request_id = request_id_of(request.scope)
    # request context vars are already cleared at this point (we sit outside the middleware),
    # so the id is passed explicitly.
    log.error("unhandled_exception", request_id=request_id, path=request.url.path, exc_info=exc)
    return problem_response(500, "internal error", request_id=request_id)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, _http_exception)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)
