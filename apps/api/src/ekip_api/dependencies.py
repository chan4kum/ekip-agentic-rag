import hashlib
import math
from collections.abc import Awaitable, Callable
from typing import Annotated

from ekip_contracts.authorization import Action, Authorizer, Resource
from ekip_contracts.chat import ChatStreamer
from ekip_contracts.identity import InvalidTokenError, SecurityContext, TokenValidator
from ekip_contracts.ratelimit import RateLimitDecision, RateLimits, RateLimitUnavailableError
from ekip_observability.context import bind_principal
from ekip_observability.logging import get_logger
from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ekip_api.services.retrieval import RetrievalService

log = get_logger("ekip.api.auth")

# auto_error=False: we want to return our own uniform 401, not FastAPI's 403 for a missing header.
_bearer = HTTPBearer(auto_error=False)
_NO_STORE = {"Cache-Control": "no-store"}


def get_retrieval_service(request: Request) -> RetrievalService:
    return RetrievalService(request.app.state.retriever)


def get_token_validator(request: Request) -> TokenValidator:
    validator: TokenValidator = request.app.state.token_validator
    return validator


def _unauthorized(*, invalid_token: bool) -> HTTPException:
    # RFC 6750: no credentials -> bare challenge; bad credentials -> error="invalid_token".
    # The client never learns WHICH check failed (the reason goes to the log only).
    challenge = 'Bearer error="invalid_token"' if invalid_token else "Bearer"
    return HTTPException(
        status_code=401,
        detail="authentication required",
        headers={"WWW-Authenticate": challenge, **_NO_STORE},
    )


async def get_security_context(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    validator: Annotated[TokenValidator, Depends(get_token_validator)],
) -> SecurityContext:
    """The ONLY way a route learns who the caller is and which tenant they belong to (I-4).

    Fails closed (I-1): anything other than a verified token is a 401; if we cannot verify
    at all (identity provider unreachable) it is a 503, never an implicit allow.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized(invalid_token=False)
    try:
        claims = await validator.validate(credentials.credentials)
    except InvalidTokenError as exc:
        if exc.reason == "key_unavailable":
            raise HTTPException(
                status_code=503, detail="authentication temporarily unavailable", headers=_NO_STORE
            ) from exc
        raise _unauthorized(invalid_token=True) from exc
    context = SecurityContext.from_claims(claims)
    # Every later log line in this request carries who/which tenant (no token, no claims dump).
    bind_principal(tenant_id=context.tenant_id, subject=context.subject)
    return context


def get_authorizer(request: Request) -> Authorizer:
    authorizer: Authorizer = request.app.state.authorizer
    return authorizer


def require_permission(
    action: Action, *, resource_kind: str = "documents"
) -> Callable[..., Awaitable[SecurityContext]]:
    """Route dependency: authenticated AND allowed to perform `action` in the caller's tenant.

    Returns the SecurityContext so a route needs only one dependency. Denied -> 403 (generic body;
    the reason goes to the log). Authorizer failure -> 503, never an implicit allow (I-1).
    """

    async def dependency(
        context: Annotated[SecurityContext, Depends(get_security_context)],
        authorizer: Annotated[Authorizer, Depends(get_authorizer)],
    ) -> SecurityContext:
        # Collection-level check: the resource is the caller's own tenant's `resource_kind`.
        # Per-document checks (Phase 5) pass the document's real owner tenant instead.
        resource = Resource(kind=resource_kind, tenant_id=context.tenant_id)
        try:
            decision = await authorizer.authorize(context, action, resource)
        except Exception as exc:
            log.error("authorizer_unavailable", error_type=type(exc).__name__)
            raise HTTPException(
                status_code=503, detail="authorization temporarily unavailable", headers=_NO_STORE
            ) from exc
        if not decision.allowed:
            log.warning(
                "authorization_denied", action=action.value, reason=decision.reason
            )  # tenant_id and subject are already bound to the log context
            raise HTTPException(status_code=403, detail="forbidden", headers=_NO_STORE)
        return context

    return dependency


def get_rate_limits(request: Request) -> RateLimits:
    limits: RateLimits = request.app.state.rate_limits
    return limits


def _headers(decision: RateLimitDecision) -> dict[str, str]:
    headers = {
        "RateLimit-Limit": str(decision.limit),
        "RateLimit-Remaining": str(decision.remaining),
    }
    if not decision.allowed:
        headers["Retry-After"] = str(max(1, math.ceil(decision.retry_after_s)))
    return headers


def limit_requests(scope: str) -> Callable[..., Awaitable[None]]:
    """Route dependency: per-user and per-tenant token buckets for `scope` (e.g. "search").

    List it AFTER require_permission in the route signature: unauthenticated or forbidden
    requests are rejected first and never consume the caller's budget.
    """

    async def dependency(
        context: Annotated[SecurityContext, Depends(get_security_context)],
        limits: Annotated[RateLimits, Depends(get_rate_limits)],
        response: Response,
    ) -> None:
        # The subject is hashed: bounded key length, and nothing from a token can shape the key.
        who = hashlib.sha256(context.subject.encode()).hexdigest()[:16]
        try:
            user = await limits.user.check(f"{context.tenant_id}:{scope}:u:{who}")
            tenant = (
                await limits.tenant.check(f"{context.tenant_id}:{scope}:t")
                if user.allowed
                else None
            )
        except RateLimitUnavailableError as exc:
            if limits.fail_open:
                log.warning("rate_limiter_unavailable_failing_open", scope=scope)
                return
            raise HTTPException(
                status_code=503,
                detail="rate limiting temporarily unavailable",
                headers={"Retry-After": "1", **_NO_STORE},
            ) from exc
        if not user.allowed:
            log.warning("rate_limited", scope=scope, bucket="user")
            raise HTTPException(429, "rate limit exceeded", headers={**_headers(user), **_NO_STORE})
        if tenant is not None and not tenant.allowed:
            log.warning("rate_limited", scope=scope, bucket="tenant")
            raise HTTPException(
                429, "rate limit exceeded", headers={**_headers(tenant), **_NO_STORE}
            )
        for name, value in _headers(user).items():
            response.headers[name] = value

    return dependency


def get_chat_streamer(request: Request) -> ChatStreamer:
    streamer: ChatStreamer = request.app.state.chat_streamer
    return streamer
