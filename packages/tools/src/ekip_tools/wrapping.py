"""Make tool output safe to put in front of a model: bounded, clean, and clearly marked as data.

This does not make injected text harmless (nothing can); it makes it recognisable, keeps it from
forging the wrapper, and caps its size. The real defence stays outside the model: the agent can
only call allowlisted tools, every call is re-authorized, and writes need a human approval.
"""

import re
import unicodedata

from ekip_contracts.tools import UNTRUSTED_CLOSE, UNTRUSTED_OPEN

_MARKER = re.compile(r"<\s*/?\s*tool_output", re.IGNORECASE)


def sanitize_text(text: str, *, max_chars: int) -> tuple[str, bool]:
    """Drop control and format characters (keep newline/tab), escape wrapper markers, cap length."""
    cleaned = "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf", "Cs", "Co", "Cn"}
    )
    cleaned = _MARKER.sub(lambda m: m.group(0).replace("<", "&lt;", 1), cleaned)
    if len(cleaned) > max_chars:
        return cleaned[:max_chars], True
    return cleaned, False


def wrap_untrusted(tool: str, text: str, *, max_chars: int) -> tuple[str, bool]:
    body, truncated = sanitize_text(text, max_chars=max_chars)
    note = ' truncated="true"' if truncated else ""
    return f'{UNTRUSTED_OPEN} tool="{tool}"{note}>\n{body}\n{UNTRUSTED_CLOSE}', truncated
