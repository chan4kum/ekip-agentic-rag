"""ASGI wiring for metrics: a pure-ASGI middleware plus a guarded /metrics endpoint.

Pure ASGI (no Starlette import) so this package stays framework-free and SSE streams are never
buffered. The /metrics endpoint is FAIL-CLOSED: it answers only when a bearer token is configured
and presented (404 otherwise - the path does not even reveal itself), or when the caller passes
`allow_anonymous=True` explicitly (local development only).
"""

import hmac
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ekip_observability.metrics import EkipMetrics

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

METRICS_PATH = "/metrics"


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


class MetricsMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: EkipMetrics,
        token: str | None = None,
        allow_anonymous: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._app = app
        self._metrics = metrics
        self._token = token or None
        self._allow_anonymous = allow_anonymous
        self._clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        if scope["path"] == METRICS_PATH:
            await self._serve_metrics(scope, send)
            return
        started = self._clock()
        status = 500  # if the app raises before responding, that is what the client sees
        self._metrics.http_in_progress.inc()

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            self._metrics.http_in_progress.dec()
            self._metrics.observe_http(
                method=str(scope["method"]),
                route=_route_template(scope),
                status=status,
                seconds=self._clock() - started,
            )
            if status == 429:
                self._metrics.rate_limit_rejections.inc()

    def _authorised(self, scope: Scope) -> bool:
        if self._allow_anonymous:
            return True
        if self._token is None:
            return False
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                presented = value.decode("latin-1")
                expected = f"Bearer {self._token}"
                return hmac.compare_digest(presented.encode(), expected.encode())
        return False

    async def _serve_metrics(self, scope: Scope, send: Send) -> None:
        if scope["method"] != "GET" or not self._authorised(scope):
            await _respond(send, 404, b"Not Found", b"text/plain")
            return
        body = generate_latest(self._metrics.registry)
        await _respond(send, 200, body, CONTENT_TYPE_LATEST.encode())


async def _respond(send: Send, status: int, body: bytes, content_type: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode()),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def install_metrics(
    app: Any,
    metrics: EkipMetrics,
    *,
    token: str | None = None,
    allow_anonymous: bool = False,
) -> None:
    """Add the middleware to a Starlette/FastAPI app (`app.add_middleware`).

    Add it OUTERMOST-ish (call it last, so it wraps routing) to time the whole request.
    """
    app.add_middleware(
        MetricsMiddleware, metrics=metrics, token=token, allow_anonymous=allow_anonymous
    )
