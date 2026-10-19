import base64
import json
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from ekip_contracts.identity import InvalidTokenError
from ekip_identity.jwks import CachedJwks
from ekip_identity.validator import JwtTokenValidator
from idp_testkit import AUDIENCE, ISSUER, claims, jwks, make_key, mint, public_jwk

KEY = make_key()
OTHER_KEY = make_key()


def _validator(*jwk_list: dict[str, Any], **kw: Any) -> JwtTokenValidator:
    async def fetch() -> dict[str, Any]:
        return dict(jwks(*jwk_list))

    return JwtTokenValidator(issuer=ISSUER, audience=AUDIENCE, keys=CachedJwks(fetch), **kw)


@pytest.fixture
def validator() -> JwtTokenValidator:
    return _validator(public_jwk(KEY, "k1"))


async def _reason(v: JwtTokenValidator, token: str) -> str:
    with pytest.raises(InvalidTokenError) as exc:
        await v.validate(token)
    return exc.value.reason


async def test_valid_token_returns_claims(validator: JwtTokenValidator) -> None:
    c = await validator.validate(mint(KEY))
    assert c.subject == "user-1"
    assert c.tenant_id == "tenant-dev"
    assert c.roles == ("ekip-user",)
    assert c.audience == (AUDIENCE,)
    assert c.authorized_party == "ekip-web"


async def test_audience_list_is_accepted_when_it_contains_ours(
    validator: JwtTokenValidator,
) -> None:
    c = await validator.validate(mint(KEY, aud=[AUDIENCE, "account"]))
    assert AUDIENCE in c.audience


async def test_expired(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, exp=1_000_000_000, iat=999_999_000)) == "expired"


async def test_not_yet_valid(validator: JwtTokenValidator) -> None:
    import time

    future = int(time.time()) + 3600
    assert await _reason(validator, mint(KEY, nbf=future)) == "not_yet_valid"


async def test_wrong_audience(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, aud="some-other-api")) == "wrong_audience"


async def test_wrong_issuer(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, iss="http://evil.test/realms/ekip")) == "wrong_issuer"


async def test_signed_by_unknown_private_key_with_known_kid(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(OTHER_KEY, kid="k1")) == "bad_signature"


async def test_tampered_payload_fails(validator: JwtTokenValidator) -> None:
    head, _payload, sig = mint(KEY).split(".")
    forged = base64.urlsafe_b64encode(json.dumps(claims(tenant_id="tenant-other")).encode()).rstrip(
        b"="
    )
    assert await _reason(validator, f"{head}.{forged.decode()}.{sig}") == "bad_signature"


async def test_alg_none_rejected(validator: JwtTokenValidator) -> None:
    def b64(d: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    token = f"{b64({'alg': 'none', 'typ': 'JWT', 'kid': 'k1'})}.{b64(claims())}."
    assert await _reason(validator, token) == "algorithm_not_allowed"


async def test_hs256_key_confusion_rejected(validator: JwtTokenValidator) -> None:
    """Classic attack: sign with HS256 using the server's PUBLIC key bytes as the HMAC secret."""
    public_pem = KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    assert await _reason(validator, _hs256(public_pem)) == "algorithm_not_allowed"


def _hs256(secret: bytes) -> str:
    import hashlib
    import hmac

    def b64(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    signing_input = (
        b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "k1"}).encode())
        + "."
        + b64(json.dumps(claims()).encode())
    )
    sig = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{b64(sig)}"


async def test_missing_kid(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, kid=None)) == "missing_kid"


async def test_overlong_kid(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, kid="k" * 500)) == "missing_kid"


async def test_unknown_kid(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, kid="nope")) == "unknown_key"


async def test_encryption_only_key_is_not_usable_for_signatures() -> None:
    v = _validator(public_jwk(KEY, "k1", use="enc", alg=None))
    assert await _reason(v, mint(KEY, kid="k1")) in {"key_unavailable", "unknown_key"}


async def test_jku_and_x5u_headers_are_ignored(validator: JwtTokenValidator) -> None:
    """A token naming its own key location must not influence which key we use."""
    token = mint(
        OTHER_KEY,
        kid="k1",
        headers={"jku": "http://evil.test/jwks.json", "x5u": "http://evil.test/c"},
    )
    assert await _reason(validator, token) == "bad_signature"


async def test_missing_tenant_claim(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, tenant_id=None)) == "missing_claim"


@pytest.mark.parametrize("bad", ["", "a b", "../x", "t" * 65, "ten ant", "a;b"])
async def test_invalid_tenant_values(validator: JwtTokenValidator, bad: str) -> None:
    assert await _reason(validator, mint(KEY, tenant_id=bad)) == "missing_claim"


async def test_non_string_tenant(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, tenant_id=123)) == "missing_claim"


@pytest.mark.parametrize("claim", ["sub", "iss", "aud", "exp", "iat"])
async def test_missing_required_claims(validator: JwtTokenValidator, claim: str) -> None:
    absent: dict[str, Any] = {claim: None}
    token = mint(KEY, **absent)
    # mint() drops None-valued overrides, so the claim is genuinely absent
    assert await _reason(validator, token) in {"missing_claim", "wrong_audience", "wrong_issuer"}


async def test_roles_must_be_list_of_strings(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, mint(KEY, roles="admin")) == "malformed"
    assert await _reason(validator, mint(KEY, roles=["ok", 5])) == "malformed"


async def test_missing_roles_means_no_roles(validator: JwtTokenValidator) -> None:
    c = await validator.validate(mint(KEY, roles=None))
    assert c.roles == ()


async def test_oversized_token(validator: JwtTokenValidator) -> None:
    assert await _reason(validator, "a" * 9000) == "too_large"


@pytest.mark.parametrize("junk", ["", "garbage", "a.b", "a.b.c", "....", "Bearer x.y.z"])
async def test_garbage_is_malformed(validator: JwtTokenValidator, junk: str) -> None:
    assert await _reason(validator, junk) == "malformed"


async def test_rejection_logs_reason_but_never_the_token(validator: JwtTokenValidator) -> None:
    from structlog.testing import capture_logs

    token = mint(KEY, aud="other")
    with capture_logs() as logs, pytest.raises(InvalidTokenError):
        await validator.validate(token)
    rendered = json.dumps(logs, default=str)
    assert "wrong_audience" in rendered
    assert token not in rendered


def test_validator_refuses_none_in_allowlist() -> None:
    async def fetch() -> dict[str, Any]:
        return {"keys": []}

    with pytest.raises(ValueError):
        JwtTokenValidator(
            issuer=ISSUER, audience=AUDIENCE, keys=CachedJwks(fetch), algorithms=("none",)
        )
    with pytest.raises(ValueError):
        JwtTokenValidator(issuer=ISSUER, audience=AUDIENCE, keys=CachedJwks(fetch), algorithms=())
