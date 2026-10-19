"""Reject oversized request bodies inside the app (defence in depth; the only limit on streams).

Traefik's `buffering` middleware enforces the limit for normal routes, but it also buffers whole
RESPONSES, which would break streaming (SSE). The chat route therefore bypasses it and relies on
this middleware instead.
"""

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ekip_api.errors import ProblemError, problem_response, request_id_of


class MaxBodySizeMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self.app = app
        self._max = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None:
            if not declared.isdigit():
                await self._reject(scope, receive, send, 400, "invalid Content-Length header")
                return
            if int(declared) > self._max:  # refuse without reading a byte of the body
                await self._reject(scope, receive, send, 413, "request body too large")
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max:  # chunked uploads have no Content-Length to check
                    raise ProblemError(413, "request body too large", slug="payload-too-large")
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
        response = problem_response(status, detail, request_id=request_id_of(scope))
        await response(scope, receive, send)
