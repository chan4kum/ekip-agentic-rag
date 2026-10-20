"""Cache key construction.

Keys are `{prefix}{layer}:{tenant}:{sha256}`. The digest covers length-prefixed (name, value)
pairs under a domain-separation header, so no two different part lists can collide by shifting
text between fields, and the raw query text never appears in Redis keys.
"""

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from ekip_contracts.documents import TENANT_ID
from ekip_contracts.retrieval import SearchScope

KEY_VERSION = "ekip.cache.v1"
DEFAULT_PREFIX = "ekip:cache:"
_LAYER = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_WS = re.compile(r"\s+")


def normalise_query(query: str) -> str:
    """NFKC + casefold + collapsed whitespace: cosmetic differences share an entry."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", query)).strip().casefold()


def scope_fingerprint(scope: SearchScope) -> str:
    """Stable id of WHO is asking (user + sorted groups). Different identities never share."""
    return _digest(KEY_VERSION, "scope", [("user", scope.user), ("groups", *sorted(scope.groups))])


@dataclass(frozen=True)
class CacheVersions:
    """Everything that changes what a correct answer is, other than the data itself."""

    retrieval_config: str
    prompt: str
    model: str
    output_policy: str

    def __post_init__(self) -> None:
        for name, value in self.as_parts():
            if not value or not value.strip():
                raise ValueError(f"cache version {name!r} must be non-empty")

    def as_parts(self) -> list[tuple[str, str]]:
        return [
            ("retrieval_config_version", self.retrieval_config),
            ("prompt_version", self.prompt),
            ("model_version", self.model),
            ("output_policy_version", self.output_policy),
        ]


def _digest(version: str, layer: str, parts: Sequence[tuple[str, ...]]) -> str:
    h = hashlib.sha256()
    for token in (version, layer):
        h.update(len(token).to_bytes(4, "big"))
        h.update(token.encode())
    for name, *values in parts:
        for token in (name, *values):
            raw = token.encode()
            h.update(len(raw).to_bytes(4, "big"))
            h.update(raw)
        h.update(b"\x00")
    return h.hexdigest()


def build_key(
    *,
    layer: str,
    tenant_id: str,
    parts: Sequence[tuple[str, str]],
    prefix: str = DEFAULT_PREFIX,
) -> str:
    if not _LAYER.fullmatch(layer):
        raise ValueError("invalid cache layer name")
    if not TENANT_ID.fullmatch(tenant_id):
        raise ValueError("invalid tenant_id")
    digest = _digest(KEY_VERSION, layer, [("tenant", tenant_id), *parts])
    return f"{prefix}{layer}:{tenant_id}:{digest}"


def document_tag(tenant_id: str, document_id: str, *, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{prefix}tag:{tenant_id}:{document_id}"


def generation_key(tenant_id: str, *, prefix: str = DEFAULT_PREFIX) -> str:
    return f"{prefix}gen:{tenant_id}"
