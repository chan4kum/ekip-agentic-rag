"""Access-token (JWT) validation. Everything here is deterministic code, never the LLM (I-2)."""

import re
from collections.abc import Sequence
from typing import Any

import jwt
from ekip_contracts.identity import AccessTokenClaims, InvalidTokenError, TokenFailure
from ekip_observability.logging import get_logger

from ekip_identity.jwks import KeyNotFoundError, KeyProvider, KeysUnavailableError

log = get_logger("ekip.identity")

_REQUIRED = ["exp", "iat", "iss", "aud", "sub"]
_TENANT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_KID = 128


def _fail(reason: TokenFailure) -> InvalidTokenError:
    return InvalidTokenError(reason)


class JwtTokenValidator:
    """Validates an OIDC access token and returns claims we may trust.

    Order matters: cheap structural checks first, then the signature, then claim checks.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        keys: KeyProvider,
        algorithms: Sequence[str] = ("RS256",),
        leeway_s: int = 10,
        max_token_chars: int = 8192,
    ) -> None:
        if not algorithms or "none" in {a.lower() for a in algorithms}:
            raise ValueError("algorithms must be a non-empty allowlist and never include 'none'")
        self._issuer = issuer
        self._audience = audience
        self._keys = keys
        self._algorithms = list(algorithms)
        self._leeway_s = leeway_s
        self._max_token_chars = max_token_chars

    async def validate(self, token: str) -> AccessTokenClaims:
        try:
            return await self._validate(token)
        except InvalidTokenError as exc:
            # Reason only; the token itself is never logged (I-8).
            log.info("token_rejected", reason=exc.reason)
            raise

    async def _validate(self, token: str) -> AccessTokenClaims:
        if len(token) > self._max_token_chars:
            raise _fail("too_large")

        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise _fail("malformed") from exc

        # 1. Algorithm allowlist: stops alg=none and RS256->HS256 "key confusion" attacks.
        #    We choose the algorithm; the token only gets to say which one it CLAIMS to use.
        if header.get("alg") not in self._algorithms:
            raise _fail("algorithm_not_allowed")

        # 2. Key lookup by kid. The kid is used only as a dictionary key: never as a path or
        #    URL. jku/x5u headers (token-supplied key locations) are ignored on purpose.
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > _MAX_KID:
            raise _fail("missing_kid")
        try:
            key = await self._keys.key_for(kid)
        except KeyNotFoundError as exc:
            raise _fail("unknown_key") from exc
        except KeysUnavailableError as exc:
            raise _fail("key_unavailable") from exc

        # 3. Signature + standard claims in one call (exp, iat, iss, aud are all REQUIRED).
        try:
            payload: dict[str, Any] = jwt.decode(
                token,
                key.key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway_s,
                options={"require": _REQUIRED},
            )
        except jwt.ExpiredSignatureError as exc:
            raise _fail("expired") from exc
        except jwt.ImmatureSignatureError as exc:
            raise _fail("not_yet_valid") from exc
        except jwt.InvalidAudienceError as exc:
            raise _fail("wrong_audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise _fail("wrong_issuer") from exc
        except jwt.InvalidSignatureError as exc:
            raise _fail("bad_signature") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise _fail("missing_claim") from exc
        except jwt.PyJWTError as exc:
            raise _fail("malformed") from exc

        return self._to_claims(payload)

    @staticmethod
    def _to_claims(payload: dict[str, Any]) -> AccessTokenClaims:
        tenant = payload.get("tenant_id")
        if not isinstance(tenant, str) or not _TENANT_ID.fullmatch(tenant):
            raise _fail("missing_claim")  # no tenant => no access (I-1, I-4)
        roles = payload.get("roles", [])
        if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
            raise _fail("malformed")
        aud = payload["aud"]
        audience = (aud,) if isinstance(aud, str) else tuple(str(a) for a in aud)
        azp = payload.get("azp")
        return AccessTokenClaims(
            subject=str(payload["sub"]),
            issuer=str(payload["iss"]),
            audience=audience,
            issued_at=int(payload["iat"]),
            expires_at=int(payload["exp"]),
            tenant_id=tenant,
            roles=tuple(roles),
            authorized_party=azp if isinstance(azp, str) else None,
        )
