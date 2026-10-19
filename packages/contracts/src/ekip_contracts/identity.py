"""Ports and types for access-token validation. Implementations live in ekip_identity."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

# Fixed, machine-readable reasons. They go to logs/audit; the client only ever sees a
# generic 401 (never "wrong audience" etc.), so attackers learn nothing about our checks.
TokenFailure = Literal[
    "malformed",
    "too_large",
    "algorithm_not_allowed",
    "missing_kid",
    "unknown_key",
    "key_unavailable",
    "bad_signature",
    "expired",
    "not_yet_valid",
    "wrong_issuer",
    "wrong_audience",
    "missing_claim",
]


class InvalidTokenError(Exception):
    """The presented token must not be trusted. `reason` is safe to log, not to show."""

    def __init__(self, reason: TokenFailure) -> None:
        super().__init__(reason)
        self.reason: TokenFailure = reason


class AccessTokenClaims(BaseModel):
    """Claims we rely on, taken ONLY from a token whose signature and context were verified."""

    model_config = ConfigDict(frozen=True)

    subject: str
    issuer: str
    audience: tuple[str, ...]
    issued_at: int
    expires_at: int
    tenant_id: str
    roles: tuple[str, ...]
    authorized_party: str | None = None


class SecurityContext(BaseModel):
    """Who is calling and for which tenant, for ONE request.

    Built only from verified claims and handed to code via a request-scoped dependency.
    Services take `tenant_id` from here, never from a request parameter, header or body (I-4).
    """

    model_config = ConfigDict(frozen=True)

    subject: str
    tenant_id: str
    roles: frozenset[str] = frozenset()
    authorized_party: str | None = None

    @classmethod
    def from_claims(cls, claims: AccessTokenClaims) -> "SecurityContext":
        return cls(
            subject=claims.subject,
            tenant_id=claims.tenant_id,
            roles=frozenset(claims.roles),
            authorized_party=claims.authorized_party,
        )

    def has_role(self, role: str) -> bool:
        return role in self.roles


class TokenValidator(Protocol):
    async def validate(self, token: str) -> AccessTokenClaims:
        """Return verified claims or raise InvalidTokenError (fail closed)."""
        ...
