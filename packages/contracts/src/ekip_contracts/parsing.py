"""Port for turning raw document bytes into structured blocks.

Parsers handle UNTRUSTED input: implementations must enforce size/page limits, never fetch
anything over the network (no SSRF through `<img src>` or links), and fail with `ParseError`
(a short, generic message) rather than leak library internals.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from ekip_contracts.documents import _require


class ParseError(Exception):
    """The document could not be parsed (corrupt, encrypted, too large, unsupported)."""


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"  # `text` holds the table as markdown
    CODE = "code"
    IMAGE = (
        "image"  # `image` holds the picture; `text` is its caption/alt text if the source has one
    )


@dataclass(frozen=True)
class ParsedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    caption: str | None = None
    alt_text: str | None = None
    bbox: tuple[float, float, float, float] | None = None  # page fractions, origin top-left

    def __post_init__(self) -> None:
        _require(len(self.data) > 0, "empty image")
        _require(self.mime_type.startswith("image/"), "not an image mime type")
        _require(self.width >= 1 and self.height >= 1, "invalid image dimensions")


@dataclass(frozen=True)
class ParsedBlock:
    kind: BlockKind
    text: str = ""
    level: int | None = None  # heading level 1..6
    page_number: int | None = None
    image: ParsedImage | None = None

    def __post_init__(self) -> None:
        _require((self.kind is BlockKind.IMAGE) == (self.image is not None), "image/kind mismatch")
        _require(
            (self.kind is BlockKind.HEADING) == (self.level is not None), "level/kind mismatch"
        )
        _require(self.level is None or 1 <= self.level <= 6, "invalid heading level")
        _require(self.kind is BlockKind.IMAGE or self.text.strip() != "", "empty text block")


@dataclass(frozen=True)
class ParsedDocument:
    blocks: tuple[ParsedBlock, ...]
    title: str | None = None
    # pages with no extractable text layer (scanned): candidates for the OCR step
    pages_needing_ocr: tuple[int, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)


class DocumentParser(Protocol):
    supported_mime_types: frozenset[str]

    async def parse(self, data: bytes, mime_type: str) -> ParsedDocument: ...
