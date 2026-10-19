"""DOCX parser (python-docx). A DOCX is a zip: sizes are checked BEFORE it is opened."""

import asyncio
import io
import zipfile

from docx import Document as load_docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from ekip_contracts.parsing import (
    BlockKind,
    ParsedBlock,
    ParsedDocument,
    ParsedImage,
    ParseError,
)

from ekip_ingestion.images import inspect_image, is_meaningful, sha256_of
from ekip_ingestion.limits import DEFAULT_LIMITS, ParseLimits
from ekip_ingestion.parsers.common import check_size, clean_text, table_to_markdown

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MAX_ZIP_ENTRIES = 10_000


class DocxParser:
    supported_mime_types = frozenset({DOCX_MIME})

    def __init__(self, limits: ParseLimits = DEFAULT_LIMITS) -> None:
        self._limits = limits

    async def parse(self, data: bytes, mime_type: str) -> ParsedDocument:
        check_size(data, self._limits)
        self._check_zip(data)
        try:
            return await asyncio.to_thread(self._parse, data)
        except ParseError:
            raise
        except Exception as exc:  # untrusted input
            raise ParseError("could not parse DOCX") from exc

    def _check_zip(self, data: bytes) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                infos = zf.infolist()
        except zipfile.BadZipFile as exc:
            raise ParseError("not a DOCX file") from exc
        if len(infos) > _MAX_ZIP_ENTRIES:
            raise ParseError("DOCX has too many entries")
        if sum(i.file_size for i in infos) > self._limits.max_zip_uncompressed_bytes:
            raise ParseError("DOCX expands beyond the allowed size")  # zip bomb

    def _parse(self, data: bytes) -> ParsedDocument:
        doc = load_docx(io.BytesIO(data))
        blocks: list[ParsedBlock] = []
        seen: set[str] = set()
        images = 0

        def add(block: ParsedBlock) -> None:
            if len(blocks) >= self._limits.max_blocks:
                raise ParseError("document has too many blocks")
            blocks.append(block)

        for child in doc.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                para = Paragraph(child, doc)
                text = clean_text(para.text)
                style = (para.style.name or "") if para.style is not None else ""
                if text:
                    add(_paragraph_block(text, style))
                for blip in child.xpath(".//a:blip"):
                    rid = blip.get(qn("r:embed"))
                    part = doc.part.related_parts.get(rid) if rid else None
                    if part is None or images >= self._limits.max_images:
                        continue
                    raw = part.blob
                    info = inspect_image(raw, self._limits)
                    if info is None or not is_meaningful(info, len(raw), self._limits):
                        continue
                    digest = sha256_of(raw)
                    if digest in seen:
                        continue
                    seen.add(digest)
                    images += 1
                    props = child.xpath(".//wp:docPr")
                    alt = clean_text(props[0].get("descr") or "") if props else ""
                    img = ParsedImage(
                        raw, info.mime_type, info.width, info.height, alt_text=alt or None
                    )
                    add(ParsedBlock(BlockKind.IMAGE, text=alt, image=img))
            elif tag == "tbl":
                table = Table(child, doc)
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                if md := table_to_markdown(rows, self._limits):
                    add(ParsedBlock(BlockKind.TABLE, md))
        title = clean_text(doc.core_properties.title or "") or next(
            (b.text for b in blocks if b.kind is BlockKind.HEADING), None
        )
        return ParsedDocument(blocks=tuple(blocks), title=title or None)


def _paragraph_block(text: str, style: str) -> ParsedBlock:
    lowered = style.lower()
    if lowered == "title":
        return ParsedBlock(BlockKind.HEADING, text, level=1)
    if lowered.startswith("heading "):
        suffix = lowered.removeprefix("heading ").strip()
        if suffix.isdigit() and 1 <= int(suffix) <= 6:
            return ParsedBlock(BlockKind.HEADING, text, level=int(suffix))
    if lowered.startswith("list"):
        return ParsedBlock(BlockKind.LIST_ITEM, text)
    return ParsedBlock(BlockKind.PARAGRAPH, text)
