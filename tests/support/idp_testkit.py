"""Test helpers: throwaway RSA keys, a minimal IdP, and token minting. Test-only code."""

import json
import time
from collections.abc import Mapping
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt.algorithms import RSAAlgorithm

ISSUER = "http://idp.test/realms/ekip"
AUDIENCE = "ekip-api"


def make_key() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def public_jwk(
    key: RSAPrivateKey, kid: str, *, use: str = "sig", alg: str | None = "RS256"
) -> dict[str, Any]:
    jwk: dict[str, Any] = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": kid, "use": use})
    if alg:
        jwk["alg"] = alg
    return jwk


def jwks(*keys: dict[str, Any]) -> Mapping[str, Any]:
    return {"keys": list(keys)}


def claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    base: dict[str, Any] = {
        "sub": "user-1",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 300,
        "tenant_id": "tenant-dev",
        "roles": ["ekip-user"],
        "azp": "ekip-web",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def mint(
    key: RSAPrivateKey,
    *,
    kid: str | None = "k1",
    alg: str = "RS256",
    headers: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> str:
    hdrs: dict[str, Any] = dict(headers or {})
    if kid is not None:
        hdrs["kid"] = kid
    return jwt.encode(claims(**overrides), key, algorithm=alg, headers=hdrs)


class TestIdp:
    """A fake identity provider: owns a key, serves its JWKS, mints tokens, builds a validator."""

    __test__ = False  # not a pytest test class

    def __init__(self, kid: str = "k1") -> None:
        self.kid = kid
        self.key = make_key()

    def validator(self) -> Any:
        from ekip_identity.jwks import CachedJwks
        from ekip_identity.validator import JwtTokenValidator

        document = jwks(public_jwk(self.key, self.kid))

        async def fetch() -> Mapping[str, Any]:
            return document

        return JwtTokenValidator(issuer=ISSUER, audience=AUDIENCE, keys=CachedJwks(fetch))

    def token(self, **overrides: Any) -> str:
        return mint(self.key, kid=self.kid, **overrides)

    def auth(self, **overrides: Any) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(**overrides)}"}


def open_limits(capacity: int = 10_000) -> Any:
    """Generous in-memory limits for tests that are not about rate limiting."""
    from ekip_contracts.ratelimit import RateLimits
    from ekip_ratelimit.memory import InMemoryRateLimiter

    return RateLimits(
        user=InMemoryRateLimiter(capacity=capacity, refill_per_second=1000),
        tenant=InMemoryRateLimiter(capacity=capacity, refill_per_second=1000),
    )


def _b64(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forge_alg_none(kid: str = "k1", **overrides: Any) -> str:
    """An unsigned token whose header claims alg=none (the classic bypass attempt)."""
    header = _b64(json.dumps({"alg": "none", "typ": "JWT", "kid": kid}).encode())
    return f"{header}.{_b64(json.dumps(claims(**overrides)).encode())}."


def forge_hs256(key: RSAPrivateKey, kid: str = "k1", **overrides: Any) -> str:
    """Key confusion: sign with HMAC-SHA256 using the server's PUBLIC key bytes as the secret."""
    import hashlib
    import hmac

    from cryptography.hazmat.primitives import serialization

    secret = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": kid}).encode())
    signing_input = f"{head}.{_b64(json.dumps(claims(**overrides)).encode())}"
    sig = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(sig)}"


def tamper_payload(token: str, **changes: Any) -> str:
    """Edit claims of a validly signed token WITHOUT re-signing (privilege escalation attempt)."""
    import base64

    head, payload, sig = token.split(".")
    data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    data.update(changes)
    return f"{head}.{_b64(json.dumps(data).encode())}.{sig}"
