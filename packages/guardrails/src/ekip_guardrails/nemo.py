"""Optional adapter: NVIDIA NeMo Guardrails behind the same GuardrailEngine port.

NeMo is NOT a dependency of this repository (it is heavy and unvetted). Install it deliberately
in the deployment image (`pip install nemoguardrails`), put a rails config directory next to the
application and construct `NemoGuardrailEngine.from_config_dir(path, base)`.

Design: the deterministic engine ALWAYS runs first and its decision can only be made stricter by
the rails, never relaxed (the classifier is not the boundary - I-2). A rails refusal reply is
treated as a block. This adapter has been exercised only against a fake rails object; verify it
against your pinned nemoguardrails version before relying on it.
"""

from typing import Any, Protocol

from ekip_contracts.guardrails import Finding, GuardrailDecision, GuardrailEngine

DEFAULT_REFUSAL = "I'm sorry, I can't respond to that."


class RailsUnavailableError(RuntimeError):
    """nemoguardrails is not installed (or the config could not be loaded)."""


class _Rails(Protocol):
    async def generate_async(self, *, messages: list[dict[str, str]]) -> Any: ...


class NemoGuardrailEngine:
    def __init__(
        self, rails: _Rails, base: GuardrailEngine, *, refusal_text: str = DEFAULT_REFUSAL
    ) -> None:
        self._rails, self._base, self._refusal = rails, base, refusal_text

    @classmethod
    def from_config_dir(cls, path: str, base: GuardrailEngine) -> "NemoGuardrailEngine":
        try:
            from nemoguardrails import LLMRails, RailsConfig  # type: ignore[import-not-found,unused-ignore]
        except ImportError as exc:  # fail with an actionable message, not a stack trace
            raise RailsUnavailableError(
                "NeMo Guardrails is not installed; run `pip install nemoguardrails` "
                "in the deployment image or use PolicyGuardrailEngine alone"
            ) from exc
        return cls(LLMRails(RailsConfig.from_path(path)), base)

    async def check_input(self, text: str, *, tenant_id: str) -> GuardrailDecision:
        base = await self._base.check_input(text, tenant_id=tenant_id)
        return await self._tighten(base, text if not base.blocked else "")

    async def check_retrieved(self, text: str, *, tenant_id: str) -> GuardrailDecision:
        # retrieved passages are screened by the deterministic engine only: sending every passage
        # through an LLM rail would be slow, costly and itself an injection surface
        return await self._base.check_retrieved(text, tenant_id=tenant_id)

    async def check_output(
        self, text: str, *, tenant_id: str, allowed_urls: frozenset[str] = frozenset()
    ) -> GuardrailDecision:
        return await self._base.check_output(text, tenant_id=tenant_id, allowed_urls=allowed_urls)

    async def _tighten(self, base: GuardrailDecision, text: str) -> GuardrailDecision:
        if base.blocked:
            return base
        try:
            reply = await self._rails.generate_async(
                messages=[{"role": "user", "content": base.text}]
            )
        except Exception:  # fail closed: a rail we cannot reach is not a pass
            return self._block(base, "engine_error")
        content = reply.get("content", "") if isinstance(reply, dict) else str(reply)
        if content.strip() == self._refusal:
            return self._block(base, "external_rail")
        return base

    @staticmethod
    def _block(base: GuardrailDecision, code: str) -> GuardrailDecision:
        extra = Finding("external_rail" if code == "external_rail" else "engine_error")
        return GuardrailDecision("block", "", (*base.findings, extra), base.policy_version)
