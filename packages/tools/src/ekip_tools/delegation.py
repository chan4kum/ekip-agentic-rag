"""Signed, short-lived tokens: delegated credentials for MCP servers, and human approvals.

Why this exists: the user's own access token must never be forwarded to tool servers (they would
be able to replay it anywhere). Instead the gateway mints a token that is
  - signed (HMAC-SHA256) with a key shared only with the tool servers,
  - bound to ONE tool (audience), ONE subject and tenant,
  - valid for seconds,
so a stolen token is useless for any other tool or later. In production the same contract is
fulfilled by an OAuth 2.0 token exchange (RFC 8693) at the identity provider; this signer is the
dependency-free implementation used in tests and single-cluster deployments.

Tokens are `base64url(payload).base64url(mac)`. The MAC covers the `purpose` claim, so a
delegation token can never be accepted as an approval and vice versa.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

from ekip_contracts.identity import SecurityContext

Purpose = Literal["delegation", "approval"]
MIN_KEY_BYTES = 32
MAX_TOKEN_CHARS = 4096
Clock = Callable[[], float]


class TokenError(Exception):
    """The token must not be trusted. `reason` is safe to log, not to show a user."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class TokenSigner:
    def __init__(self, key: bytes, *, clock: Clock = time.time) -> None:
        if len(key) < MIN_KEY_BYTES:
            raise ValueError(f"signing key must be at least {MIN_KEY_BYTES} bytes")
        self._key = key
        self._clock = clock

    def now(self) -> int:
        return int(self._clock())

    def mint(self, purpose: Purpose, claims: Mapping[str, Any], *, ttl_s: int) -> str:
        now = self.now()
        payload = {
            **claims,
            "purpose": purpose,
            "iat": now,
            "exp": now + ttl_s,
            "jti": secrets.token_hex(12),
        }
        body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        return f"{body}.{_b64(self._mac(body))}"

    def verify(self, token: str, purpose: Purpose) -> dict[str, Any]:
        if not token or len(token) > MAX_TOKEN_CHARS or token.count(".") != 1:
            raise TokenError("malformed")
        body, mac = token.split(".")
        try:
            given = _unb64(mac)
        except ValueError as exc:
            raise TokenError("malformed") from exc
        if not hmac.compare_digest(given, self._mac(body)):
            raise TokenError("bad_signature")
        try:
            payload = json.loads(_unb64(body))
        except ValueError as exc:
            raise TokenError("malformed") from exc
        if not isinstance(payload, dict) or payload.get("purpose") != purpose:
            raise TokenError("wrong_purpose")
        exp = payload.get("exp")
        if not isinstance(exp, int) or exp <= self.now():
            raise TokenError("expired")
        return payload

    def _mac(self, body: str) -> bytes:
        return hmac.new(self._key, body.encode("ascii"), hashlib.sha256).digest()


def mint_delegation(
    signer: TokenSigner, context: SecurityContext, tool: str, *, ttl_s: int = 30
) -> str:
    """A token that lets ONE tool server act as `context` for the next `ttl_s` seconds."""
    return signer.mint(
        "delegation",
        {
            "sub": context.subject,
            "tenant": context.tenant_id,
            "roles": sorted(context.roles),
            "principal": context.principal,
            "groups": sorted(context.groups),
            "azp": context.authorized_party,
            "aud": tool,
        },
        ttl_s=ttl_s,
    )


def verify_delegation(signer: TokenSigner, token: str, tool: str) -> SecurityContext:
    """Server side: rebuild the caller's SecurityContext from a delegation token for `tool`."""
    claims = signer.verify(token, "delegation")
    if claims.get("aud") != tool:
        raise TokenError("wrong_audience")
    try:
        return SecurityContext(
            subject=str(claims["sub"]),
            tenant_id=str(claims["tenant"]),
            roles=frozenset(claims.get("roles") or ()),
            principal=claims.get("principal"),
            groups=frozenset(claims.get("groups") or ()),
            authorized_party=claims.get("azp"),
        )
    except (KeyError, ValueError) as exc:
        raise TokenError("missing_claim") from exc
