import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from ekip_contracts.authorization import Action
from ekip_contracts.chat import ChatStreamer
from ekip_contracts.identity import SecurityContext
from ekip_observability.logging import get_logger
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ekip_api.dependencies import get_chat_streamer, limit_requests, require_permission
from ekip_api.errors import request_id_of
from ekip_api.schemas import ChatRequest
from ekip_api.sse import HEARTBEAT, StreamDeadlineError, format_event, with_heartbeats

router = APIRouter(tags=["chat"])
log = get_logger("ekip.api.chat")

# Streaming responses must not be cached, transformed or buffered by any hop.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",  # tells nginx-style proxies not to buffer
}


@router.post("/api/v1/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    ctx: Annotated[SecurityContext, Depends(require_permission(Action.DOCUMENTS_SEARCH))],
    _rate_limited: Annotated[None, Depends(limit_requests("chat"))],
    streamer: Annotated[ChatStreamer, Depends(get_chat_streamer)],
) -> StreamingResponse:
    """Authentication, authorization and rate limiting all run BEFORE the stream starts, so those
    failures are ordinary HTTP errors (401/403/429 problem+json), never a half-open stream."""
    settings = request.app.state.settings
    request_id = request_id_of(request.scope)
    conversation_id = body.conversation_id or uuid.uuid4().hex

    async def events() -> AsyncIterator[bytes]:
        yield format_event("meta", {"request_id": request_id, "conversation_id": conversation_id})
        try:
            async for item in with_heartbeats(
                streamer.stream(message=body.message, context=ctx),
                interval_s=settings.chat_keepalive_s,
                total_timeout_s=settings.chat_timeout_s,
            ):
                if isinstance(item, bytes):
                    yield HEARTBEAT
                else:
                    yield format_event(item.event, item.data)
            yield format_event("done", {"finish_reason": "stop"})
        except StreamDeadlineError:
            log.warning("chat_stream_deadline_exceeded", timeout_s=settings.chat_timeout_s)
            yield _error("timeout", "the answer took too long", request_id)
        except Exception as exc:  # headers are already sent: report in-band, never raise
            log.error("chat_stream_failed", error_type=type(exc).__name__, exc_info=exc)
            yield _error("internal-error", "internal error", request_id)
        # asyncio.CancelledError (client disconnect) is deliberately not caught: it must propagate

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


def _error(slug: str, detail: str, request_id: str) -> bytes:
    return format_event(
        "error",
        {"type": f"urn:ekip:problem:{slug}", "detail": detail, "request_id": request_id},
    )
