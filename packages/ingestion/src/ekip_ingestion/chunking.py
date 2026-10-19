"""Structure-aware, deterministic chunking. No model calls.

Rules (see the learning log for the reasoning):
- A chunk never crosses a heading: section boundaries are chunk boundaries.
- Paragraphs/list items are packed up to `target_chars`; a table or code block is its own chunk.
- Oversized pieces are split at the gentlest boundary available (table rows with the header
  repeated, code at line breaks, prose at sentences with a small overlap).
- Same input => same chunk ids and hashes (so unchanged chunks can reuse embeddings).
- An image becomes an ImageAsset (+ an `image` chunk if it has any text) in document order.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

from ekip_contracts.assets import ImageAsset
from ekip_contracts.documents import (
    MAX_CHUNK_CHARS,
    MAX_SECTION_DEPTH,
    Chunk,
    ChunkKind,
    ContractError,
    Document,
    sha256_hex,
)
from ekip_contracts.parsing import BlockKind, ParsedBlock, ParsedDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PAGE_NUMBER_ONLY = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.I)


@dataclass(frozen=True)
class ChunkingConfig:
    target_chars: int = 2_000  # about 500 tokens
    max_chars: int = 3_200  # about 800 tokens; hard ceiling for any chunk we emit
    overlap_chars: int = 200  # only when splitting ONE long paragraph

    def __post_init__(self) -> None:
        if not (200 <= self.target_chars <= self.max_chars <= MAX_CHUNK_CHARS):
            raise ContractError("need 200 <= target_chars <= max_chars <= MAX_CHUNK_CHARS")
        if not (0 <= self.overlap_chars < self.target_chars // 2):
            raise ContractError("overlap_chars must be < target_chars / 2")


@dataclass(frozen=True)
class ImageEnrichment:
    """Text produced for an image by OCR / a vision model (looked up by the image's sha256)."""

    ocr_text: str | None = None
    description: str | None = None
    model_version: str | None = None


@dataclass(frozen=True)
class PendingAsset:
    asset: ImageAsset
    data: bytes  # to be written to the ObjectStore under asset.sha256


@dataclass(frozen=True)
class ChunkingResult:
    chunks: tuple[Chunk, ...]
    assets: tuple[PendingAsset, ...]


def strip_boilerplate(parsed: ParsedDocument) -> ParsedDocument:
    """Drop running headers/footers and bare page numbers.

    A short paragraph that repeats on at least 3 pages (and at least half of the pages that have
    text) is page furniture, not content. Removing it keeps it out of the index entirely.
    """
    pages_with_text = {b.page_number for b in parsed.blocks if b.page_number is not None}
    seen_on: dict[str, set[int]] = {}
    for b in parsed.blocks:
        if b.kind is BlockKind.PARAGRAPH and b.page_number is not None and len(b.text) <= 120:
            seen_on.setdefault(b.text, set()).add(b.page_number)
    threshold = max(3, (len(pages_with_text) + 1) // 2)
    furniture = {t for t, pages in seen_on.items() if len(pages) >= threshold}

    def keep(b: ParsedBlock) -> bool:
        if b.kind is not BlockKind.PARAGRAPH:
            return True
        return b.text not in furniture and not (
            b.page_number is not None and _PAGE_NUMBER_ONLY.fullmatch(b.text.strip())
        )

    kept = tuple(b for b in parsed.blocks if keep(b))
    return ParsedDocument(kept, parsed.title, parsed.pages_needing_ocr, parsed.warnings)


def embedding_text(document: Document, chunk: Chunk) -> str:
    """What is sent to the embedding model: title and section path give a chunk its context
    deterministically (no LLM 'contextual retrieval' call). Stored content stays unchanged."""
    path = " > ".join(chunk.section_path)
    header = f"{document.title} > {path}" if path else document.title
    return f"{header}\n{chunk.content}"


def chunk_document(
    document: Document,
    parsed: ParsedDocument,
    config: ChunkingConfig = ChunkingConfig(),  # noqa: B008 - frozen dataclass
    enrichments: Mapping[str, ImageEnrichment] | None = None,
) -> ChunkingResult:
    return _Chunker(document, config, enrichments or {}).run(strip_boilerplate(parsed))


class _Chunker:
    def __init__(
        self, document: Document, config: ChunkingConfig, enrichments: Mapping[str, ImageEnrichment]
    ) -> None:
        self.doc = document
        self.cfg = config
        self.enrich = enrichments
        self.chunks: list[Chunk] = []
        self.assets: list[PendingAsset] = []
        self.headings: list[tuple[int, str]] = []  # (level, text) stack
        self.buf: list[str] = []  # blocks waiting to be packed
        self.buf_len = 0
        self.buf_page: int | None = None

    # ---- driver ---------------------------------------------------------------------------
    def run(self, parsed: ParsedDocument) -> ChunkingResult:
        for block in parsed.blocks:
            match block.kind:
                case BlockKind.HEADING:
                    self._flush()
                    self._push_heading(block)
                case BlockKind.PARAGRAPH | BlockKind.LIST_ITEM:
                    self._add_text(block)
                case BlockKind.TABLE:
                    self._flush()
                    for piece in _split_table(block.text, self.cfg.max_chars):
                        self._emit(piece, block.page_number)
                case BlockKind.CODE:
                    self._flush()
                    for piece in _split_lines(block.text, self.cfg.max_chars):
                        self._emit(piece, block.page_number)
                case BlockKind.IMAGE:
                    self._flush()
                    self._image(block)
        self._flush()
        return ChunkingResult(tuple(self.chunks), tuple(self.assets))

    # ---- text packing ---------------------------------------------------------------------
    def _add_text(self, block: ParsedBlock) -> None:
        sep = "\n" if block.kind is BlockKind.LIST_ITEM and self.buf else "\n\n"
        if len(block.text) > self.cfg.max_chars:
            self._flush()
            for piece in _split_sentences(block.text, self.cfg):
                self._emit(piece, block.page_number)
            return
        extra = len(sep) if self.buf else 0
        if self.buf and self.buf_len + extra + len(block.text) > self.cfg.target_chars:
            self._flush()
            sep = ""
        if not self.buf:
            self.buf_page = block.page_number
            sep = ""
        self.buf.append(sep + block.text)
        self.buf_len += len(sep) + len(block.text)

    def _flush(self) -> None:
        if self.buf:
            self._emit("".join(self.buf), self.buf_page)
        self.buf, self.buf_len, self.buf_page = [], 0, None

    def _emit(
        self,
        content: str,
        page: int | None,
        *,
        kind: ChunkKind = ChunkKind.TEXT,
        asset_sha256: str | None = None,
    ) -> None:
        if not content.strip():
            return
        self.chunks.append(
            Chunk.create(
                document=self.doc,
                ordinal=len(self.chunks),
                content=content,
                section_path=self._section_path(),
                page_number=page,
                kind=kind,
                asset_sha256=asset_sha256,
            )
        )

    # ---- structure ------------------------------------------------------------------------
    def _push_heading(self, block: ParsedBlock) -> None:
        level = block.level or 1
        while self.headings and self.headings[-1][0] >= level:
            self.headings.pop()
        self.headings.append((level, block.text[:256]))

    def _section_path(self) -> tuple[str, ...]:
        return tuple(text for _, text in self.headings)[-MAX_SECTION_DEPTH:]

    # ---- images ---------------------------------------------------------------------------
    def _image(self, block: ParsedBlock) -> None:
        image = block.image
        if image is None:  # impossible by ParsedBlock's invariant; keeps mypy and readers honest
            return
        sha = sha256_hex(image.data)
        extra = self.enrich.get(sha)
        asset = ImageAsset(
            tenant_id=self.doc.tenant_id,
            document_id=self.doc.document_id,
            document_version=self.doc.version,
            ordinal=len(self.assets),
            sha256=sha,
            mime_type=image.mime_type,
            byte_size=len(image.data),
            width=image.width,
            height=image.height,
            page_number=block.page_number,
            bbox=image.bbox,
            nearby_chunk_ordinal=len(self.chunks) - 1 if self.chunks else None,
            caption=image.caption,
            alt_text=image.alt_text,
            ocr_text=extra.ocr_text if extra else None,
            description=extra.description if extra else None,
            description_model_version=extra.model_version if extra else None,
        )
        self.assets.append(PendingAsset(asset, image.data))
        if text := asset.search_text:
            self._emit(
                text[: self.cfg.max_chars],
                block.page_number,
                kind=ChunkKind.IMAGE,
                asset_sha256=sha,
            )


# ---- splitters (pure functions) ------------------------------------------------------------
def _split_table(markdown: str, limit: int) -> list[str]:
    if len(markdown) <= limit:
        return [markdown]
    lines = markdown.split("\n")
    header, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    current: list[str] = list(header)
    size = sum(len(x) + 1 for x in current)
    for row in rows:
        row = row[: limit - size_of(header) - 1]  # a single absurd row cannot exceed the limit
        if size + len(row) + 1 > limit and len(current) > len(header):
            pieces.append("\n".join(current))
            current, size = list(header), size_of(header)
        current.append(row)
        size += len(row) + 1
    if len(current) > len(header):
        pieces.append("\n".join(current))
    return pieces


def size_of(lines: list[str]) -> int:
    return sum(len(x) + 1 for x in lines)


def _split_lines(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        for part in [line[i : i + limit] for i in range(0, len(line), limit)] or [""]:
            if size + len(part) + 1 > limit and current:
                pieces.append("\n".join(current))
                current, size = [], 0
            current.append(part)
            size += len(part) + 1
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]


def _split_sentences(text: str, cfg: ChunkingConfig) -> list[str]:
    """Pack sentences up to target_chars; repeat the tail of each piece (<= overlap) in the next."""
    sentences: list[str] = []
    for s in _SENTENCE_END.split(text):
        # a "sentence" longer than the ceiling (no punctuation) is cut at whitespace
        while len(s) > cfg.max_chars:
            cut = s.rfind(" ", 0, cfg.max_chars)
            cut = cut if cut > cfg.max_chars // 2 else cfg.max_chars
            sentences.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            sentences.append(s)
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for s in sentences:
        if current and size + len(s) + 1 > cfg.target_chars:
            pieces.append(" ".join(current))
            tail: list[str] = []
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) + 1 > cfg.overlap_chars:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            current, size = tail, tail_size
        current.append(s)
        size += len(s) + 1
    if current and (not pieces or " ".join(current) != pieces[-1]):
        pieces.append(" ".join(current))
    return pieces
