from dataclasses import replace
from datetime import UTC, datetime

import pytest
from ekip_contracts.documents import ContractError, DocumentAcl, sha256_hex
from ekip_contracts.events import ChangeKind, Lane, SourceChange

T0 = datetime(2026, 10, 19, 6, 0, tzinfo=UTC)
ACL = DocumentAcl("acl-1", frozenset({"alice"}), frozenset({"eng"}))


def upsert(**kw: object) -> SourceChange:
    base: dict[str, object] = {
        "event_id": "evt-1", "tenant_id": "tenant-a", "source": "sharepoint",
        "source_document_id": "site/doc.docx", "kind": ChangeKind.UPSERT, "occurred_at": T0,
        "source_version": "7", "content_hash": sha256_hex("body"), "mime_type": "text/html",
        "title": "Manual", "source_uri": "https://example.test/doc", "acl": ACL,
    }  # fmt: skip
    return SourceChange(**{**base, **kw})  # type: ignore[arg-type]


def acl_change(**kw: object) -> SourceChange:
    base: dict[str, object] = {
        "event_id": "evt-2", "tenant_id": "tenant-a", "source": "sharepoint",
        "source_document_id": "site/doc.docx", "kind": ChangeKind.ACL_CHANGE,
        "occurred_at": T0, "acl": ACL,
    }  # fmt: skip
    return SourceChange(**{**base, **kw})  # type: ignore[arg-type]


def delete(**kw: object) -> SourceChange:
    base: dict[str, object] = {
        "event_id": "evt-3", "tenant_id": "tenant-a", "source": "sharepoint",
        "source_document_id": "site/doc.docx", "kind": ChangeKind.DELETE, "occurred_at": T0,
    }  # fmt: skip
    return SourceChange(**{**base, **kw})  # type: ignore[arg-type]


@pytest.mark.parametrize("make", [upsert, acl_change, delete])
def test_events_round_trip_through_json(make) -> None:  # type: ignore[no-untyped-def]
    e = make()
    assert SourceChange.from_json(e.to_json()) == e


def test_redelivery_has_the_same_idempotency_key_but_a_real_change_does_not() -> None:
    assert upsert().idempotency_key == upsert(event_id="evt-99").idempotency_key  # redelivery
    assert upsert().idempotency_key != upsert(source_version="8").idempotency_key
    assert upsert().idempotency_key != upsert(content_hash=sha256_hex("other")).idempotency_key
    assert upsert().idempotency_key != upsert(tenant_id="tenant-b").idempotency_key
    assert delete().idempotency_key != delete(occurred_at=T0.replace(hour=9)).idempotency_key
    assert upsert().idempotency_key != delete().idempotency_key


def test_a_document_keeps_one_partition_key_across_event_kinds() -> None:
    assert upsert().partition_key == acl_change().partition_key == delete().partition_key
    assert upsert().partition_key != upsert(source_document_id="other").partition_key
    assert upsert().partition_key != upsert(tenant_id="tenant-b").partition_key


@pytest.mark.parametrize(
    "bad",
    [
        lambda: upsert(content_hash=None),
        lambda: upsert(acl=None),
        lambda: upsert(content_hash="nothex"),
        lambda: upsert(mime_type="not a mime"),
        lambda: upsert(occurred_at=datetime(2026, 1, 1)),  # naive
        lambda: upsert(tenant_id="bad tenant!"),
        lambda: acl_change(acl=None),
        lambda: acl_change(content_hash=sha256_hex("x")),
        lambda: delete(acl=ACL),
        lambda: delete(title="x"),
    ],
)
def test_invalid_events_are_rejected_at_construction(bad) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ContractError):
        bad()


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[]",
        b"{}",
        b'{"kind":"upsert"}',
        b'{"event_id":1}',
        b"null",
        b"x" * 2_000_000,
    ],
)
def test_malformed_wire_messages_raise_contract_error_only(raw: bytes) -> None:
    with pytest.raises(ContractError):
        SourceChange.from_json(raw)


def test_lane_defaults_to_live_and_survives_the_wire() -> None:
    assert upsert().lane is Lane.LIVE
    e = replace(upsert(), lane=Lane.BACKFILL)
    assert SourceChange.from_json(e.to_json()).lane is Lane.BACKFILL
