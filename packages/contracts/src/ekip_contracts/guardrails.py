"""Port for content guardrails (PII, secrets, prompt-injection, exfiltration, prompt leaks).

Guardrails are DEFENCE IN DEPTH. They never replace authorization: the LLM and any classifier are
not a security boundary (invariant I-2). A decision carries only fixed finding codes and counts -
never the text that triggered it - so it is safe to log and audit.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

GuardrailAction = Literal["allow", "mask", "block"]

FindingCode = Literal[
    "input_too_long",
    "control_chars",
    "input_role_spoof",
    "pii_email",
    "pii_phone",
    "pii_card",
    "pii_iban",
    "pii_ssn_us",
    "pii_aadhaar",
    "pii_pan_in",
    "secret_credential",
    "prompt_leak",
    "injection_marker",
    "external_url",
    "engine_error",
    "external_rail",
]

_SEVERITY: dict[str, int] = {"allow": 0, "mask": 1, "block": 2}


def stricter(a: GuardrailAction, b: GuardrailAction) -> GuardrailAction:
    """The more restrictive of two actions (block > mask > allow)."""
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


@dataclass(frozen=True)
class Finding:
    code: FindingCode
    count: int = 1


@dataclass(frozen=True)
class GuardrailDecision:
    """What to do with a piece of text.

    `text` is safe to use as-is: the original (allow), the redacted version (mask), or "" (block),
    so a caller that forgets to check `action` still cannot leak a blocked text.
    """

    action: GuardrailAction
    text: str
    findings: tuple[Finding, ...] = ()
    policy_version: str = ""

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(f.code for f in self.findings)

    @property
    def blocked(self) -> bool:
        return self.action == "block"


class GuardrailEngine(Protocol):
    """Checks text at the three trust boundaries. Implementations must be deterministic for the
    same policy and must raise nothing for ordinary text; callers treat any exception as a block
    (fail closed)."""

    async def check_input(self, text: str, *, tenant_id: str) -> GuardrailDecision:
        """A user question, before it reaches a model."""
        ...

    async def check_retrieved(self, text: str, *, tenant_id: str) -> GuardrailDecision:
        """A retrieved passage (untrusted data), before it is placed in a prompt."""
        ...

    async def check_output(
        self, text: str, *, tenant_id: str, allowed_urls: frozenset[str] = frozenset()
    ) -> GuardrailDecision:
        """A model reply, before the user or any later node sees it. URLs in `allowed_urls`
        (the citation sources) may stay; every other external URL is an exfiltration channel."""
        ...
