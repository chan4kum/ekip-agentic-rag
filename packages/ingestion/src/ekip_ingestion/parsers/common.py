"""Helpers shared by the parsers."""

import re
import unicodedata

from ekip_contracts.parsing import ParseError

from ekip_ingestion.limits import ParseLimits

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0]+")


def clean_text(text: str) -> str:
    """NFC-normalize, drop control and zero-width characters, collapse runs of spaces."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", _CONTROL.sub("", text))
    lines = (_SPACES.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n"))
    return "\n".join(lines).strip()


def check_size(data: bytes, limits: ParseLimits) -> None:
    if not data:
        raise ParseError("empty document")
    if len(data) > limits.max_document_bytes:
        raise ParseError("document too large")


def table_to_markdown(rows: list[list[str]], limits: ParseLimits) -> str:
    """Render rows as a GitHub-style markdown table (first row is the header)."""
    rows = [[clean_text(c).replace("|", "\\|").replace("\n", " ") for c in r] for r in rows if r]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    if width * len(rows) > limits.max_table_cells:
        raise ParseError("table too large")
    padded = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join(lines)
