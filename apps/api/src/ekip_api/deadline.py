"""Per-request deadline: cancel a handler that runs too long and answer 504.

Why: without a bound, one stuck dependency (a slow search, a hung LLM call) holds a worker, a
connection and the caller's patience forever. The deadline must be SHORTER than the proxy's
(Traefik responseHeader timeout 60 s) so the app answers with a proper problem+json before the
edge gives up with an opaque 502/504.
"""

import asyncio

from ekip_observability.logging import get_logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ekip_api.errors import problem_response, request_id_of

log = get_logger("ekip.api.deadline")


class RequestDeadlineMiddleware:
    def __init__(
        self, app: ASGIApp, *, timeout_s: float, exempt_prefixes: tuple[str, ...] = ()
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.app = app
        self._timeout_s = timeout_s
        self._exempt = exempt_prefixes  # e.g. streaming routes that manage their own deadline

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].startswith(self._exempt):
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        deadline = asyncio.timeout(self._timeout_s)
        try:
            async with deadline:
                await self.app(scope, receive, tracking_send)
        except TimeoutError:
            if not deadline.expired():
                raise  # a TimeoutError raised by the handler itself is the handler's problem
            log.warning(
                "request_deadline_exceeded",
                path=scope["path"],
                timeout_s=self._timeout_s,
                response_started=response_started,
            )
            if response_started:
                return  # headers are out; all we can do is stop (the client sees a cut stream)
            response = problem_response(
                504, "the request took too long", request_id=request_id_of(scope)
            )
            await response(scope, receive, tracking_send)
