"""Query hygiene: one place that decides what text is allowed to reach the search engine."""

import re
import unicodedata

MAX_QUERY_CHARS = 1000
_WHITESPACE = re.compile(r"\s+")


class InvalidQueryError(ValueError):
    """The caller supplied input the retriever refuses to act on."""


def normalize_query(raw: str, *, max_chars: int = MAX_QUERY_CHARS) -> str:
    """NFKC-normalise, turn control characters into spaces, collapse whitespace, then bound it.

    Normalising first means "ＡＭＭ-32" and "AMM-32" search the same, and a query cannot hide
    its real length behind invisible characters.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(" " if unicodedata.category(ch).startswith("C") else ch for ch in text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        raise InvalidQueryError("query must not be empty")
    if len(text) > max_chars:
        raise InvalidQueryError("query too long")
    return text
