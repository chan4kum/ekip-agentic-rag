"""Human approval for write tools.

A write tool never runs on the model's say-so. The flow is:
  1. The agent asks for the tool; the gateway answers APPROVAL_REQUIRED with an `ApprovalRequest`
     that names the tool and a digest of the EXACT arguments.
  2. A human (a different person than the requester, holding an approver role) reviews it and the
     approval service issues a signed grant bound to that request.
  3. The agent calls again with the grant. The gateway re-computes the digest from the arguments
     in THIS call, so changing even one character after approval invalidates the grant.
  4. Each grant works once (replay guard) and expires quickly.

The service is transport-agnostic: Lane A's graph persists the request in its checkpoint
(interrupt), the UI shows `summary`, and the approver's decision calls `issue`.
"""

import hashlib
import json
import secrets
from collections.abc import Mapping
from typing import Any, Protocol

from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ApprovalRequest, ToolSpec

from ekip_tools.delegation import TokenError, TokenSigner

DEFAULT_TTL_S = 900


class ApprovalError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def arguments_digest(arguments: Mapping[str, Any]) -> str:
    """Stable fingerprint of the arguments (sorted keys, no whitespace)."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ReplayGuard(Protocol):
    async def first_use(self, token_id: str, expires_at: int) -> bool:
        """True exactly once per `token_id` (atomic). Redis SET NX in production."""
        ...


class InMemoryReplayGuard:
    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    async def first_use(self, token_id: str, expires_at: int) -> bool:
        if token_id in self._seen:
            return False
        self._seen[token_id] = expires_at
        return True


class ApprovalService:
    def __init__(
        self,
        signer: TokenSigner,
        replay: ReplayGuard,
        *,
        approver_roles: frozenset[str] = frozenset({"ekip-admin"}),
        ttl_s: int = DEFAULT_TTL_S,
        allow_self_approval: bool = False,
    ) -> None:
        self._signer = signer
        self._replay = replay
        self._approver_roles = approver_roles
        self._ttl_s = ttl_s
        self._allow_self = allow_self_approval

    def request(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=secrets.token_hex(8),
            tool=spec.name,
            arguments_digest=arguments_digest(arguments),
            requested_by=context.subject,
            tenant_id=context.tenant_id,
            summary=f"Run '{spec.name}' ({spec.description[:120]}) as {context.subject}",
            expires_at=self._signer.now() + self._ttl_s,
        )

    def issue(self, request: ApprovalRequest, approver: SecurityContext) -> str:
        """Called by the approval UI/endpoint after a human said yes. Fails closed."""
        if self._signer.now() >= request.expires_at:
            raise ApprovalError("request_expired")
        if approver.tenant_id != request.tenant_id:
            raise ApprovalError("cross_tenant")
        if not approver.roles.intersection(self._approver_roles):
            raise ApprovalError("not_an_approver")
        if approver.subject == request.requested_by and not self._allow_self:
            raise ApprovalError("self_approval")
        return self._signer.mint(
            "approval",
            {
                "rid": request.request_id,
                "tool": request.tool,
                "digest": request.arguments_digest,
                "requester": request.requested_by,
                "tenant": request.tenant_id,
                "approver": approver.subject,
            },
            ttl_s=max(1, request.expires_at - self._signer.now()),
        )

    async def consume(
        self,
        token: str,
        context: SecurityContext,
        spec: ToolSpec,
        arguments: Mapping[str, Any],
    ) -> tuple[str, str]:
        """Check the grant against THIS call and burn it. Returns (request_id, approver)."""
        try:
            claims = self._signer.verify(token, "approval")
        except TokenError as exc:
            raise ApprovalError(exc.reason) from exc
        if (
            claims.get("tool") != spec.name
            or claims.get("tenant") != context.tenant_id
            or claims.get("requester") != context.subject
        ):
            raise ApprovalError("grant_mismatch")
        if not secrets.compare_digest(str(claims.get("digest")), arguments_digest(arguments)):
            raise ApprovalError("arguments_changed")
        if not await self._replay.first_use(str(claims["jti"]), int(claims["exp"])):
            raise ApprovalError("already_used")
        return str(claims["rid"]), str(claims["approver"])
