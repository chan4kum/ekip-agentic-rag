"""Plain text and Markdown. No external dependency: the grammar we need is small."""

import asyncio
import re

from ekip_contracts.parsing import BlockKind, ParsedBlock, ParsedDocument, ParseError

from ekip_ingestion.limits import DEFAULT_LIMITS, ParseLimits
from ekip_ingestion.parsers.common import check_size, clean_text

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^(```|~~~)")
_LIST = re.compile(r"^\s{0,6}(?:[-*+]|\d{1,9}[.)])[ \t]+(.+)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


class TextParser:
    supported_mime_types = frozenset({"text/plain", "text/markdown"})

    def __init__(self, limits: ParseLimits = DEFAULT_LIMITS) -> None:
        self._limits = limits

    async def parse(self, data: bytes, mime_type: str) -> ParsedDocument:
        check_size(data, self._limits)
        return await asyncio.to_thread(self._parse, data, mime_type)

    def _parse(self, data: bytes, mime_type: str) -> ParsedDocument:
        text = data.decode("utf-8-sig", errors="replace")
        blocks = _markdown(text) if mime_type == "text/markdown" else _plain(text)
        if len(blocks) > self._limits.max_blocks:
            raise ParseError("document has too many blocks")
        title = next((b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 1), None)
        return ParsedDocument(blocks=tuple(blocks), title=title)


def _plain(text: str) -> list[ParsedBlock]:
    out: list[ParsedBlock] = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        cleaned = clean_text(para).replace("\n", " ")
        if cleaned:
            out.append(ParsedBlock(BlockKind.PARAGRAPH, cleaned))
    return out


def _markdown(text: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    lines = text.replace("\r\n", "\n").split("\n")
    para: list[str] = []
    i = 0

    def flush() -> None:
        joined = clean_text(" ".join(para))
        if joined:
            blocks.append(ParsedBlock(BlockKind.PARAGRAPH, joined))
        para.clear()

    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line.strip()):
            flush()
            fence = line.strip()[:3]
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith(fence):
                code.append(lines[i])
                i += 1
            body = "\n".join(code).strip("\n")
            if body.strip():
                blocks.append(ParsedBlock(BlockKind.CODE, body))
            i += 1  # closing fence (or end of input)
            continue
        if (m := _HEADING.match(line)) and clean_text(m.group(2)):
            flush()
            blocks.append(
                ParsedBlock(BlockKind.HEADING, clean_text(m.group(2)), level=len(m.group(1)))
            )
        elif (
            line.lstrip().startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1])
        ):
            flush()
            rows = [line]
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            blocks.append(ParsedBlock(BlockKind.TABLE, "\n".join(r.strip() for r in rows)))
            continue
        elif m := _LIST.match(line):
            flush()
            item = clean_text(m.group(1))
            if item:
                blocks.append(ParsedBlock(BlockKind.LIST_ITEM, item))
        elif not line.strip():
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return blocks
