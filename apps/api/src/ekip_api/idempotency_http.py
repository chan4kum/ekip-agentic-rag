"""HTTP glue for idempotent POST routes (header parsing, replay response, error mapping)."""

import re
from collections.abc import Awaitable, Callable
from typing import Annotated

from ekip_contracts.idempotency import StoredResponse
from fastapi import Header, HTTPException, Request, Response

from ekip_api.errors import ProblemError
from ekip_api.services.idempotency import (
    IdempotencyInProgressError,
    IdempotencyMismatchError,
    IdempotencyService,
)

_KEY = re.compile(r"^[A-Za-z0-9_.:\-]{8,128}$")


def get_idempotency(request: Request) -> IdempotencyService:
    service: IdempotencyService = request.app.state.idempotency
    return service


def idempotency_key(
    value: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> str:
    """Required on opted-in routes: a retry without a key could double-apply the operation."""
    if value is None:
        raise HTTPException(400, "Idempotency-Key header is required")
    if not _KEY.fullmatch(value):
        raise HTTPException(400, "Idempotency-Key must be 8-128 characters of A-Z a-z 0-9 _ . : -")
    return value


def to_response(stored: StoredResponse, *, replayed: bool) -> Response:
    return Response(
        content=stored.body,
        status_code=stored.status,
        media_type=stored.content_type,
        headers={"Idempotent-Replayed": "true"} if replayed else None,
    )


async def run_idempotent(
    service: IdempotencyService,
    *,
    scope: str,
    key: str,
    fingerprint: str,
    produce: Callable[[], Awaitable[StoredResponse]],
) -> Response:
    """Call the service and translate its domain errors into HTTP problems."""
    try:
        stored, replayed = await service.run(
            scope=scope, key=key, fingerprint=fingerprint, produce=produce
        )
    except IdempotencyInProgressError as exc:
        raise ProblemError(
            409,
            "a request with this Idempotency-Key is still in progress",
            slug="idempotency-in-progress",
            headers={"Retry-After": "1"},
        ) from exc
    except IdempotencyMismatchError as exc:
        raise ProblemError(
            422,
            "Idempotency-Key was already used with a different request",
            slug="idempotency-key-reused",
        ) from exc
    return to_response(stored, replayed=replayed)
