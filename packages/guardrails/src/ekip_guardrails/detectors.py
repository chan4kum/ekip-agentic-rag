"""Pure detectors. Each returns spans or counts - never the matched text - and runs on text that
was first stripped of invisible characters and NFKC-normalized, so look-alike and zero-width
tricks cannot hide a pattern."""

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from ekip_contracts.guardrails import FindingCode

# zero-width, bidi controls/overrides, word joiner, BOM, soft hyphen; C0/C1 controls except \t \n \r
_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\u00ad]")
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def clean(text: str) -> tuple[str, int]:
    """Strip invisible/control characters, then NFKC-normalize. Returns (text, removed_count)."""
    without = _CONTROL.sub("", _INVISIBLE.sub("", text))
    return unicodedata.normalize("NFKC", without), len(text) - len(without)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    code: FindingCode
    replacement: str


# ---- PII -------------------------------------------------------------------------------------
def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    rearranged = value[4:] + value[:4]
    number = "".join(str(int(c, 36)) for c in rearranged)
    return int(number) % 97 == 1


_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_ok(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_SSN = re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])")
_AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_PAN = re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w-])\+?\d[\d ().\-]{8,17}\d(?![\w-])")


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _card(m: re.Match[str]) -> bool:
    d = _digits(m.group())
    return 13 <= len(d) <= 19 and luhn_ok(d)


def _iban(m: re.Match[str]) -> bool:
    v = m.group().replace(" ", "")
    return 15 <= len(v) <= 34 and iban_ok(v)


def _aadhaar(m: re.Match[str]) -> bool:
    return verhoeff_ok(_digits(m.group()))


def _phone(m: re.Match[str]) -> bool:
    d = _digits(m.group())
    return 10 <= len(d) <= 15 and not re.fullmatch(r"\d{4}[-.]\d{2}[-.]\d{2}", m.group())


# (code, locale, regex, validator, replacement) in priority order: earlier wins an overlap
_PII: tuple[tuple[FindingCode, str, re.Pattern[str], Callable[[re.Match[str]], bool], str], ...] = (
    ("pii_card", "global", _CARD, _card, "[REDACTED:CARD]"),
    ("pii_iban", "global", _IBAN, _iban, "[REDACTED:IBAN]"),
    ("pii_ssn_us", "us", _SSN, lambda m: True, "[REDACTED:SSN]"),
    ("pii_aadhaar", "in", _AADHAAR, _aadhaar, "[REDACTED:AADHAAR]"),
    ("pii_pan_in", "in", _PAN, lambda m: True, "[REDACTED:PAN]"),
    ("pii_email", "global", _EMAIL, lambda m: True, "[REDACTED:EMAIL]"),
    ("pii_phone", "global", _PHONE, _phone, "[REDACTED:PHONE]"),
)


def spans_free(spans: list[Span], start: int, end: int) -> bool:
    return all(end <= s.start or start >= s.end for s in spans)


def find_pii(text: str, locales: Iterable[str]) -> list[Span]:
    wanted = set(locales)
    spans: list[Span] = []
    for code, locale, pattern, valid, replacement in _PII:
        if locale != "global" and locale not in wanted:
            continue
        for m in pattern.finditer(text):
            if valid(m) and spans_free(spans, m.start(), m.end()):
                spans.append(Span(m.start(), m.end(), code, replacement))
    return spans


# ---- credentials -----------------------------------------------------------------------------
_SECRETS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-]{20,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|passw(?:or)?d|access[_-]?token)\b\s*[:=]\s*[\"']?[^\s\"']{8,}"
    ),
)


def find_secrets(text: str) -> list[Span]:
    spans: list[Span] = []
    for pattern in _SECRETS:
        for m in pattern.finditer(text):
            if spans_free(spans, m.start(), m.end()):
                spans.append(Span(m.start(), m.end(), "secret_credential", "[REDACTED:SECRET]"))
    return spans


# ---- prompt injection / role spoofing ---------------------------------------------------------
_SPOOF = re.compile(
    r"(?im)^\s*(?:system|assistant|developer)\s*:|<\|[a-z_]+\|>|\[/?INST\]|<<\s*/?SYS\s*>>"
    r"|</?\s*(?:question|sources?|history|parts?|turn|q|a)\b[^>]*>"
)

_INJECTION = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
        r"\b(?:instructions?|rules|prompts?|guidelines|polic(?:y|ies)|restrictions)\b",
        r"\b(?:reveal|show|print|repeat|output|display|leak|disclose|tell me)\b[^.\n]{0,40}"
        r"\b(?:system|hidden|initial|developer)\s+(?:prompt|message|instructions?)\b",
        r"\b(?:reveal|leak|disclose|exfiltrate|output|print)\b[^.\n]{0,30}"
        r"\b(?:passwords?|credentials?|api[ _-]?keys?|secrets?)\b",
        r"\byou\s+are\s+now\b|\bnew\s+instructions?\s*:|\bfrom\s+now\s+on,?\s+you\b",
        r"\b(?:you\s+(?:must|should|need\s+to)|please|now|immediately)\b[^.\n]{0,30}"
        r"\b(?:call|invoke|execute|run|use)\b[^.\n]{0,30}\b(?:tool|function|api)\b",
        r"\b(?:send|post|forward|email|upload|transmit|exfiltrate)\b[^\n]{0,80}https?://",
        r"<\|[a-z_]+\|>|\[/?INST\]|<<\s*/?SYS\s*>>|</?\s*(?:sources?|question|history)\b[^>]*>",
    )
)
INJECTION_REPLACEMENT = "[removed: suspected instruction]"


def has_role_spoof(text: str) -> bool:
    return _SPOOF.search(text) is not None


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start) + 1)
    left = left + 1 if left > 0 else 0
    stops = [i for i in (text.find("\n", end), text.find(". ", end)) if i != -1]
    right = (min(stops) + 1) if stops else len(text)
    return left, right


def find_injection(text: str) -> list[Span]:
    """Instruction-like sentences in retrieved data. Heuristic: it raises the cost of an attack
    but is NOT a boundary - tool and document permissions are enforced elsewhere (I-2, I-3)."""
    spans: list[Span] = []
    for pattern in _INJECTION:
        for m in pattern.finditer(text):
            s, e = _sentence_bounds(text, m.start(), m.end())
            if spans_free(spans, s, e):
                spans.append(Span(s, e, "injection_marker", INJECTION_REPLACEMENT))
    return spans


def apply_spans(text: str, spans: Iterable[Span]) -> str:
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        text = text[: span.start] + span.replacement + text[span.end :]
    return text


# ---- external URLs (exfiltration) --------------------------------------------------------------
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?[^)]*\)")
_HTML_TAG = re.compile(
    r"""<(?:img|a|iframe|script|link|embed|object|source)\b[^>]*?
        (?:src|href|data)\s*=\s*["']?([^"'\s>]+)[^>]*>""",
    re.IGNORECASE | re.VERBOSE,
)
_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*<?([^)\s>]+)>?[^)]*\)")
_REF_DEF = re.compile(r"(?m)^[ \t]*\[[^\]]+\]:[ \t]*(\S+).*$")
_BARE_URL = re.compile(r"(?<![\w:/])(?:https?://|//)[^\s<>\"'`)\]]+", re.IGNORECASE)
_TRAILING = ".,;:!?"
URL_REMOVED = "[link removed]"


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(_TRAILING)
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return (
        f"{parts.scheme.lower()}://{host}{parts.path}{('?' + parts.query) if parts.query else ''}"
    )


def is_external(url: str) -> bool:
    u = url.strip()
    internal = u.startswith(("#", "./", "../")) or (u.startswith("/") and not u.startswith("//"))
    return not internal


def strip_external_urls(
    text: str, allowed: frozenset[str] = frozenset(), *, images_only: bool = False
) -> tuple[str, int]:
    """Remove external images, links and bare URLs that are not in `allowed`. Returns the new text
    and how many were removed. With `images_only`, only image-like embeds are touched (documents
    legitimately contain plain links; auto-loading images are the exfiltration channel)."""
    ok = {normalize_url(a) for a in allowed}
    removed = 0

    def bad(url: str) -> bool:
        return is_external(url) and normalize_url(url) not in ok

    def image(m: re.Match[str]) -> str:
        nonlocal removed
        if bad(m.group(1)):
            removed += 1
            return ""
        return m.group()

    def html_tag(m: re.Match[str]) -> str:
        nonlocal removed
        tag = m.group().lstrip("<").split(None, 1)[0].lower().rstrip(">")
        if images_only and tag not in {"img", "iframe", "script", "embed", "object", "source"}:
            return m.group()
        if bad(m.group(1)):
            removed += 1
            return ""
        return m.group()

    text = _MD_IMAGE.sub(image, text)
    text = _HTML_TAG.sub(html_tag, text)
    if images_only:
        return text, removed

    def link(m: re.Match[str]) -> str:
        nonlocal removed
        if bad(m.group(2)):
            removed += 1
            return f"[{m.group(1)}]"
        return m.group()

    def ref(m: re.Match[str]) -> str:
        nonlocal removed
        if bad(m.group(1)):
            removed += 1
            return ""
        return m.group()

    def bare(m: re.Match[str]) -> str:
        nonlocal removed
        if normalize_url(m.group()) in ok:
            return m.group()
        removed += 1
        return URL_REMOVED

    text = _MD_LINK.sub(link, text)
    text = _REF_DEF.sub(ref, text)
    text = _BARE_URL.sub(bare, text)
    return text, removed
