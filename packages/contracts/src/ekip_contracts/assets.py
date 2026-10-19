"""Image assets extracted from documents: bytes live in the object store, metadata in Postgres.

An image is made searchable by a text chunk (`ChunkKind.IMAGE`) whose content is the asset's
`search_text` (caption + alt text + OCR + description), so retrieval treats it like any other chunk
and the answer can point back to the asset.
"""

import re
from dataclasses import dataclass

from ekip_contracts.documents import (
    DOCUMENT_ID,
    MODEL_VERSION,
    SHA256_HEX,
    TENANT_ID,
    _require,
)

MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_TEXT_CHARS = 5_000
MAX_IMAGE_DIMENSION = 20_000
_IMAGE_MIME = re.compile(r"^image/[a-z0-9][a-z0-9.+-]*$")


def _check_text(value: str | None, name: str) -> None:
    if value is None:
        return
    _require(len(value) <= MAX_IMAGE_TEXT_CHARS, f"{name} too long")
    _require("\x00" not in value, f"{name} contains a NUL byte")


@dataclass(frozen=True)
class ImageAsset:
    tenant_id: str
    document_id: str
    document_version: int
    ordinal: int  # position among the images of this document version
    sha256: str  # of the image bytes == the object-store key
    mime_type: str
    byte_size: int
    width: int
    height: int
    page_number: int | None = None
    # (x0, y0, x1, y1) as fractions of the page, origin top-left; None if unknown
    bbox: tuple[float, float, float, float] | None = None
    nearby_chunk_ordinal: int | None = None  # text chunk the image sits next to
    caption: str | None = None  # from the document
    alt_text: str | None = None  # from the document
    ocr_text: str | None = None  # text inside the image
    description: str | None = None  # model-generated; provenance in description_model_version
    description_model_version: str | None = None

    def __post_init__(self) -> None:
        _require(TENANT_ID.fullmatch(self.tenant_id), "invalid tenant_id")
        _require(DOCUMENT_ID.fullmatch(self.document_id), "invalid document_id")
        _require(self.document_version >= 1 and self.ordinal >= 0, "invalid version/ordinal")
        _require(SHA256_HEX.fullmatch(self.sha256), "invalid sha256")
        _require(_IMAGE_MIME.fullmatch(self.mime_type), "invalid image mime_type")
        _require(1 <= self.byte_size <= MAX_IMAGE_BYTES, "byte_size out of range")
        _require(1 <= self.width <= MAX_IMAGE_DIMENSION, "width out of range")
        _require(1 <= self.height <= MAX_IMAGE_DIMENSION, "height out of range")
        _require(self.page_number is None or self.page_number >= 1, "page_number must be >= 1")
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            _require(
                0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0, "bbox must be a box inside 0..1"
            )
        _require(
            self.nearby_chunk_ordinal is None or self.nearby_chunk_ordinal >= 0,
            "invalid nearby_chunk_ordinal",
        )
        for name in ("caption", "alt_text", "ocr_text", "description"):
            _check_text(getattr(self, name), name)
        _require(
            self.description_model_version is None
            or MODEL_VERSION.fullmatch(self.description_model_version),
            "invalid description_model_version",
        )
        _require(
            self.description is None or self.description_model_version is not None,
            "a model-generated description needs its description_model_version",
        )

    @property
    def search_text(self) -> str:
        """What gets chunked, embedded and searched for this image ('' if nothing is known)."""
        parts = (self.caption, self.alt_text, self.ocr_text, self.description)
        return "\n".join(p.strip() for p in parts if p and p.strip())
