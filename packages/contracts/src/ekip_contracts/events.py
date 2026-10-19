"""Change events: what connectors tell the platform about a source document.

Events travel over a log (Kafka) that delivers AT LEAST ONCE and may reorder across partitions, so
every consumer must be idempotent. Two properties make that possible:
- `idempotency_key`: equal for redeliveries of the same logical change, different otherwise.
- `occurred_at`: when the change happened AT THE SOURCE; a change older than what we already hold is
  stale and is ignored, whatever order it arrives in.

Events are small. An UPSERT does not carry file bytes: the connector first puts them in the object
store (content-addressed) and the event names them by `content_hash`.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from ekip_contracts.documents import (
    MAX_TITLE_CHARS,
    PRINCIPAL,
    SHA256_HEX,
    SOURCE_NAME,
    TENANT_ID,
    ContractError,
    DocumentAcl,
    _require,
    _require_aware,
    _require_text,
    document_id_for,
    sha256_hex,
)

MAX_EVENT_BYTES = 1_048_576
_MIME = re.compile(r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")


class ChangeKind(StrEnum):
    UPSERT = "upsert"  # created or content changed (an ACL may come with it)
    ACL_CHANGE = "acl_change"  # permissions changed, content did not: no re-embedding
    DELETE = "delete"


class Lane(StrEnum):
    """Priority lane. A million-document backfill must never starve live updates or revocations."""

    LIVE = "live"
    BACKFILL = "backfill"


@dataclass(frozen=True)
class SourceChange:
    event_id: str
    tenant_id: str
    source: str
    source_document_id: str
    kind: ChangeKind
    occurred_at: datetime
    lane: Lane = Lane.LIVE
    # UPSERT only
    source_version: str | None = None
    content_hash: str | None = None
    mime_type: str | None = None
    title: str | None = None
    source_uri: str | None = None
    # UPSERT and ACL_CHANGE
    acl: DocumentAcl | None = None

    def __post_init__(self) -> None:
        _require(PRINCIPAL.fullmatch(self.event_id), "invalid event_id")
        _require(TENANT_ID.fullmatch(self.tenant_id), "invalid tenant_id")
        _require(SOURCE_NAME.fullmatch(self.source), "invalid source")
        _require_text(self.source_document_id, 512, "source_document_id")
        _require_aware(self.occurred_at, "occurred_at")
        upsert_fields = (
            self.source_version,
            self.content_hash,
            self.mime_type,
            self.title,
            self.source_uri,
        )
        if self.kind is ChangeKind.UPSERT:
            _require(all(f is not None for f in upsert_fields), "UPSERT needs its content fields")
            _require(self.acl is not None, "UPSERT needs an ACL")
            _require_text(self.source_version or "", 128, "source_version")
            _require(
                bool(self.content_hash and SHA256_HEX.fullmatch(self.content_hash)),
                "invalid content_hash",
            )
            _require(bool(self.mime_type and _MIME.fullmatch(self.mime_type)), "invalid mime_type")
            _require_text(self.title or "", MAX_TITLE_CHARS, "title")
            _require_text(self.source_uri or "", 2048, "source_uri")
        else:
            _require(
                all(f is None for f in upsert_fields),
                f"{self.kind.value} carries no content fields",
            )
            _require(
                (self.acl is not None) == (self.kind is ChangeKind.ACL_CHANGE),
                "ACL only on ACL_CHANGE",
            )

    @property
    def document_id(self) -> str:
        return document_id_for(self.tenant_id, self.source, self.source_document_id)

    @property
    def partition_key(self) -> str:
        """Same key => same Kafka partition => one document's events stay in order."""
        return f"{self.tenant_id}:{self.document_id}"

    @property
    def idempotency_key(self) -> str:
        parts: tuple[str | None, ...]
        if self.kind is ChangeKind.UPSERT:
            parts = (self.source_version, self.content_hash)
        elif self.kind is ChangeKind.ACL_CHANGE:
            parts = (self.acl.acl_version if self.acl else "", self.occurred_at.isoformat())
        else:
            parts = (self.occurred_at.isoformat(),)
        material = "|".join(
            (self.tenant_id, self.document_id, self.kind.value, *(p or "" for p in parts))
        )
        return sha256_hex(material)

    # ---- wire format ------------------------------------------------------------------------

    def to_json(self) -> str:
        body: dict[str, Any] = {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "source": self.source,
            "source_document_id": self.source_document_id,
            "kind": self.kind.value,
            "lane": self.lane.value,
            "occurred_at": self.occurred_at.isoformat(),
            "source_version": self.source_version,
            "content_hash": self.content_hash,
            "mime_type": self.mime_type,
            "title": self.title,
            "source_uri": self.source_uri,
            "acl": None
            if self.acl is None
            else {
                "acl_version": self.acl.acl_version,
                "allowed_users": sorted(self.acl.allowed_users),
                "allowed_groups": sorted(self.acl.allowed_groups),
            },
        }
        return json.dumps(body, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "SourceChange":
        """Parse an untrusted message. Raises ContractError (never KeyError/TypeError)."""
        if len(raw) > MAX_EVENT_BYTES:
            raise ContractError("event too large")
        try:
            body = json.loads(raw)
            acl = body.get("acl")
            return cls(
                event_id=body["event_id"],
                tenant_id=body["tenant_id"],
                source=body["source"],
                source_document_id=body["source_document_id"],
                kind=ChangeKind(body["kind"]),
                lane=Lane(body.get("lane", "live")),
                occurred_at=datetime.fromisoformat(body["occurred_at"]),
                source_version=body.get("source_version"),
                content_hash=body.get("content_hash"),
                mime_type=body.get("mime_type"),
                title=body.get("title"),
                source_uri=body.get("source_uri"),
                acl=None
                if acl is None
                else DocumentAcl(
                    acl_version=acl["acl_version"],
                    allowed_users=frozenset(acl["allowed_users"]),
                    allowed_groups=frozenset(acl["allowed_groups"]),
                ),
            )
        except ContractError:
            raise
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ContractError("malformed change event") from exc
