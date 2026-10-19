"""HTML parser (BeautifulSoup, stdlib html.parser: no JavaScript, no network, no CSS engine).

Security: scripts/styles/frames/svg are dropped; text a human cannot see (hidden attribute,
aria-hidden, display:none, visibility:hidden, zero font size or opacity, comments) is dropped too,
because hidden text is the classic carrier for indirect prompt injection (invariant I-3). Images
are only taken from inline `data:` URIs; `src="http://..."` is never fetched (no SSRF).
"""

import asyncio
import base64
import binascii
import re

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag
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

_DROP = {
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "object",
    "embed",
    "svg",
    "canvas",
    "head",
    "link",
    "meta",
    "audio",
    "video",
    "form",
    "button",
    "select",
    "textarea",
}
_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_PARA = {"p", "blockquote", "dt", "dd", "figcaption", "summary", "caption", "address"}
_INLINE = {
    "a",
    "abbr",
    "b",
    "bdi",
    "bdo",
    "br",
    "cite",
    "code",
    "data",
    "dfn",
    "em",
    "i",
    "kbd",
    "mark",
    "q",
    "s",
    "samp",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "time",
    "u",
    "var",
    "wbr",
    "del",
    "ins",
}
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?\s*(?:;|$)"
    r"|opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)|(?:^|;)\s*height\s*:\s*0(?:px)?\s*;[^\"]*overflow\s*:\s*hidden",
    re.I,
)


def _is_hidden(tag: Tag) -> bool:
    aria = tag.get("aria-hidden")
    if tag.has_attr("hidden") or (isinstance(aria, str) and aria.lower() == "true"):
        return True
    style = tag.get("style")
    return isinstance(style, str) and bool(_HIDDEN_STYLE.search(style))


class HtmlParser:
    supported_mime_types = frozenset({"text/html", "application/xhtml+xml"})

    def __init__(self, limits: ParseLimits = DEFAULT_LIMITS) -> None:
        self._limits = limits

    async def parse(self, data: bytes, mime_type: str) -> ParsedDocument:
        check_size(data, self._limits)
        return await asyncio.to_thread(self._parse, data)

    def _parse(self, data: bytes) -> ParsedDocument:
        soup = BeautifulSoup(data.decode("utf-8-sig", errors="replace"), "html.parser")
        title_tag = soup.find("title")
        title = clean_text(title_tag.get_text()) if title_tag else None
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            c.extract()
        for hidden in [t for t in soup.find_all(True) if _is_hidden(t)]:
            hidden.decompose()
        for name in _DROP - {"head"}:
            for t in soup.find_all(name):
                t.decompose()
        walker = _Walker(self._limits)
        walker.walk(soup.body or soup)
        walker.flush()
        if title is None:
            title = next((b.text for b in walker.blocks if b.kind is BlockKind.HEADING), None)
        warnings = (
            (f"{walker.skipped_external_images} external image(s) not fetched",)
            if walker.skipped_external_images
            else ()
        )
        return ParsedDocument(blocks=tuple(walker.blocks), title=title or None, warnings=warnings)


class _Walker:
    def __init__(self, limits: ParseLimits) -> None:
        self.limits = limits
        self.blocks: list[ParsedBlock] = []
        self.pending: list[str] = []
        self.skipped_external_images = 0
        self._images = 0
        self._seen: set[str] = set()

    def _add(self, block: ParsedBlock) -> None:
        if len(self.blocks) >= self.limits.max_blocks:
            raise ParseError("document has too many blocks")
        self.blocks.append(block)

    def flush(self) -> None:
        text = clean_text(" ".join(self.pending))
        self.pending.clear()
        if text:
            self._add(ParsedBlock(BlockKind.PARAGRAPH, text))

    def walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                self.pending.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if name in _INLINE:
                self.pending.append(child.get_text(" "))
            elif name in _HEADINGS:
                self.flush()
                if text := clean_text(child.get_text(" ")):
                    self._add(ParsedBlock(BlockKind.HEADING, text, level=_HEADINGS[name]))
            elif name in _PARA:
                self.flush()
                if text := clean_text(child.get_text(" ")):
                    self._add(ParsedBlock(BlockKind.PARAGRAPH, text))
            elif name == "li":
                self.flush()
                own = [
                    c for c in child.children if not (isinstance(c, Tag) and c.name in ("ul", "ol"))
                ]
                text = clean_text(
                    " ".join(c.get_text(" ") if isinstance(c, Tag) else str(c) for c in own)
                )
                if text:
                    self._add(ParsedBlock(BlockKind.LIST_ITEM, text))
                for nested in child.find_all(["ul", "ol"], recursive=False):
                    self.walk(nested)
            elif name == "pre":
                self.flush()
                if (code := child.get_text().strip("\n")).strip():
                    self._add(ParsedBlock(BlockKind.CODE, code))
            elif name == "table":
                self.flush()
                rows = [
                    [c.get_text(" ") for c in tr.find_all(["th", "td"])]
                    for tr in child.find_all("tr")
                ]
                if md := table_to_markdown(rows, self.limits):
                    self._add(ParsedBlock(BlockKind.TABLE, md))
            elif name == "img":
                self.flush()
                self._image(child)
            else:  # div, section, ul, ol, figure, ...: containers
                self.flush()
                self.walk(child)
                self.flush()

    def _image(self, tag: Tag) -> None:
        src = tag.get("src")
        if not (
            isinstance(src, str) and src.lower().startswith("data:image/") and ";base64," in src
        ):
            self.skipped_external_images += 1
            return
        if self._images >= self.limits.max_images:
            return
        b64 = src.split(";base64,", 1)[1]
        if len(b64) > self.limits.max_image_bytes * 4 // 3 + 4:
            return
        try:
            raw = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError):
            return
        info = inspect_image(raw, self.limits)
        if info is None or not is_meaningful(info, len(raw), self.limits):
            return
        digest = sha256_of(raw)
        if digest in self._seen:
            return
        self._seen.add(digest)
        self._images += 1
        alt = clean_text(str(tag.get("alt") or "")) or None
        caption = None
        fig = tag.find_parent("figure")
        if fig is not None and (cap := fig.find("figcaption")):
            caption = clean_text(cap.get_text(" ")) or None
        image = ParsedImage(
            raw, info.mime_type, info.width, info.height, caption=caption, alt_text=alt
        )
        self._add(ParsedBlock(BlockKind.IMAGE, text=caption or alt or "", image=image))
