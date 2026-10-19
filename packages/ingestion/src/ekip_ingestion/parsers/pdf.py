"""PDF parser (pypdf: pure Python, no JavaScript execution, no external renderer).

Text comes from the text layer. A page with no text layer is reported in `pages_needing_ocr`
so the OCR step runs ONLY where it is needed; we never OCR a text PDF.
"""

import asyncio
import io
import re

from ekip_contracts.parsing import (
    BlockKind,
    ParsedBlock,
    ParsedDocument,
    ParsedImage,
    ParseError,
)
from pypdf import PdfReader

from ekip_ingestion.images import inspect_image, is_meaningful, sha256_of
from ekip_ingestion.limits import DEFAULT_LIMITS, ParseLimits
from ekip_ingestion.parsers.common import check_size, clean_text

_PARA_BREAK = re.compile(r"\n\s*\n")
_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")


class PdfParser:
    supported_mime_types = frozenset({"application/pdf"})

    def __init__(self, limits: ParseLimits = DEFAULT_LIMITS) -> None:
        self._limits = limits

    async def parse(self, data: bytes, mime_type: str) -> ParsedDocument:
        check_size(data, self._limits)
        if not data.lstrip()[:5] == b"%PDF-":
            raise ParseError("not a PDF")
        try:
            return await asyncio.to_thread(self._parse, data)
        except ParseError:
            raise
        except Exception as exc:  # untrusted input: any library failure is a parse failure
            raise ParseError("could not parse PDF") from exc

    def _parse(self, data: bytes) -> ParsedDocument:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise ParseError("encrypted PDFs are not supported")
        if len(reader.pages) > self._limits.max_pages:
            raise ParseError("PDF has too many pages")
        blocks: list[ParsedBlock] = []
        needs_ocr: list[int] = []
        warnings: list[str] = []
        seen: set[str] = set()
        images = repeated = 0
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\r\n", "\n")
            text = _HYPHEN_WRAP.sub(r"\1\2", text)
            paragraphs = [clean_text(p).replace("\n", " ") for p in _PARA_BREAK.split(text)]
            paragraphs = [p for p in paragraphs if p]
            if not paragraphs:
                needs_ocr.append(number)
            blocks.extend(
                ParsedBlock(BlockKind.PARAGRAPH, p, page_number=number) for p in paragraphs
            )
            for img in self._page_images(page):
                if images >= self._limits.max_images:
                    break
                info = inspect_image(img, self._limits)
                if info is None or not is_meaningful(info, len(img), self._limits):
                    continue
                digest = sha256_of(img)
                if digest in seen:
                    repeated += 1  # e.g. a logo repeated on every page: keep the first only
                    continue
                seen.add(digest)
                images += 1
                parsed = ParsedImage(img, info.mime_type, info.width, info.height)
                blocks.append(ParsedBlock(BlockKind.IMAGE, image=parsed, page_number=number))
            if len(blocks) > self._limits.max_blocks:
                raise ParseError("document has too many blocks")
        if needs_ocr:
            warnings.append(f"{len(needs_ocr)} page(s) without a text layer")
        if repeated:
            warnings.append(f"{repeated} repeated image(s) skipped")
        meta = reader.metadata
        title = clean_text(str(meta.title)) if meta is not None and meta.title else None
        return ParsedDocument(
            blocks=tuple(blocks),
            title=title or None,
            pages_needing_ocr=tuple(needs_ocr),
            warnings=tuple(warnings),
        )

    def _page_images(self, page: object) -> list[bytes]:
        try:
            return [i.data for i in page.images]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a broken image must not fail the whole document
            return []
