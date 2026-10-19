"""Rules both repository adapters apply identically."""

from collections.abc import Sequence

from ekip_contracts.assets import ImageAsset
from ekip_contracts.documents import Chunk, ContractError


def check_chunks(tenant_id: str, document_id: str, version: int, chunks: Sequence[Chunk]) -> None:
    """Every chunk belongs to exactly this tenant/document/version, with no duplicate ordinals."""
    seen: set[int] = set()
    for c in chunks:
        if (c.tenant_id, c.document_id, c.document_version) != (tenant_id, document_id, version):
            raise ContractError("chunk does not belong to this document version")
        if c.ordinal in seen:
            raise ContractError("duplicate chunk ordinal")
        seen.add(c.ordinal)


def check_assets(
    tenant_id: str, document_id: str, version: int, assets: Sequence[ImageAsset]
) -> None:
    seen: set[int] = set()
    for a in assets:
        if (a.tenant_id, a.document_id, a.document_version) != (tenant_id, document_id, version):
            raise ContractError("asset does not belong to this document version")
        if a.ordinal in seen:
            raise ContractError("duplicate asset ordinal")
        seen.add(a.ordinal)
