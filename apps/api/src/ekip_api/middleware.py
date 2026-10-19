import re
import time

from ekip_observability.context import bind_request_context, clear_request_context, new_request_id
from ekip_observability.logging import get_logger
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
log = get_logger("ekip.http")


def _request_id(incoming: str | None) -> str:
    # Only accept well-formed IDs; anything else is replaced (prevents log injection)
    return incoming if incoming and _SAFE_ID.fullmatch(incoming) else new_request_id()


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _request_id(Headers(scope=scope).get("x-request-id"))
        bind_request_context(request_id)
        started, status = time.perf_counter(), 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log.info(
                "http_request",
                method=scope["method"],
                path=scope["path"],
                client_ip=scope["client"][0] if scope.get("client") else None,
                status=status,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            clear_request_context()
