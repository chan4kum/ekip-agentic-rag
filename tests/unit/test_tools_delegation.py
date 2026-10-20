import pytest
from ekip_tools import TokenError, TokenSigner, mint_delegation, verify_delegation
from tools_testkit import ALICE, KEY, FakeClock


def test_roundtrip_rebuilds_the_security_context() -> None:
    signer = TokenSigner(KEY)
    token = mint_delegation(signer, ALICE, "search_documents")
    ctx = verify_delegation(signer, token, "search_documents")
    assert (ctx.subject, ctx.tenant_id, ctx.principal) == ("u-alice", "tenant-a", "alice")
    assert ctx.roles == ALICE.roles


def test_token_is_bound_to_one_tool() -> None:
    signer = TokenSigner(KEY)
    token = mint_delegation(signer, ALICE, "search_documents")
    with pytest.raises(TokenError, match="wrong_audience"):
        verify_delegation(signer, token, "query_database")


def test_token_expires() -> None:
    clock = FakeClock()
    signer = TokenSigner(KEY, clock=clock)
    token = mint_delegation(signer, ALICE, "search_documents", ttl_s=30)
    clock.advance(31)
    with pytest.raises(TokenError, match="expired"):
        verify_delegation(signer, token, "search_documents")


def test_tampering_and_foreign_keys_are_rejected() -> None:
    signer = TokenSigner(KEY)
    token = mint_delegation(signer, ALICE, "search_documents")
    body, mac = token.split(".")
    with pytest.raises(TokenError, match="bad_signature"):
        verify_delegation(signer, f"{body}x.{mac}", "search_documents")
    with pytest.raises(TokenError, match="bad_signature"):
        verify_delegation(TokenSigner(b"z" * 32), token, "search_documents")
    for junk in ("", "abc", "a.b.c", "x" * 5000):
        with pytest.raises(TokenError):
            verify_delegation(signer, junk, "search_documents")


def test_a_delegation_cannot_be_used_as_an_approval() -> None:
    signer = TokenSigner(KEY)
    token = mint_delegation(signer, ALICE, "search_documents")
    with pytest.raises(TokenError, match="wrong_purpose"):
        signer.verify(token, "approval")


def test_short_keys_are_refused() -> None:
    with pytest.raises(ValueError, match="at least"):
        TokenSigner(b"short")
