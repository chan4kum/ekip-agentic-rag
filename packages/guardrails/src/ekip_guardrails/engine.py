"""The deterministic guardrail engine. Same text + same policy = same decision, no model, no
network, so it is unit-testable and cannot be talked out of a rule."""

from collections import Counter

from ekip_contracts.guardrails import (
    Finding,
    FindingCode,
    GuardrailAction,
    GuardrailDecision,
    stricter,
)

from ekip_guardrails import detectors
from ekip_guardrails.policy import GuardrailPolicy, PolicySet, normalize

Staged = tuple[list[detectors.Span], GuardrailAction]


class _Run:
    """Accumulates findings and the strictest action for one check."""

    def __init__(self, policy: GuardrailPolicy, text: str) -> None:
        self.policy = policy
        self.text = text
        self.action: GuardrailAction = "allow"
        self.counts: Counter[FindingCode] = Counter()

    def add(self, code: FindingCode, action: GuardrailAction, count: int = 1) -> None:
        if count > 0 and action != "allow":
            self.counts[code] += count
            self.action = stricter(self.action, action)

    def apply(self, *stages: Staged) -> None:
        """Record findings for spans (all computed on the CURRENT text) and rewrite the text once.
        Stages are in priority order: a span overlapping an earlier one is ignored."""
        chosen: list[detectors.Span] = []
        to_mask: list[detectors.Span] = []
        for spans, action in stages:
            if action == "allow":
                continue
            for span in spans:
                if detectors.spans_free(chosen, span.start, span.end):
                    chosen.append(span)
                    self.add(span.code, action)
                    if action == "mask":
                        to_mask.append(span)
        if to_mask:
            self.text = detectors.apply_spans(self.text, to_mask)

    def decision(self, original: str) -> GuardrailDecision:
        findings = tuple(Finding(code, n) for code, n in sorted(self.counts.items()))
        text = {"allow": original, "mask": self.text, "block": ""}[self.action]
        return GuardrailDecision(self.action, text, findings, self.policy.version)


class PolicyGuardrailEngine:
    def __init__(self, policies: PolicySet | GuardrailPolicy | None = None) -> None:
        if policies is None:
            policies = PolicySet()
        elif isinstance(policies, GuardrailPolicy):
            policies = PolicySet(default=policies)
        self._policies = policies

    async def check_input(self, text: str, *, tenant_id: str) -> GuardrailDecision:
        policy = self._policies.for_tenant(tenant_id)
        cleaned, removed = detectors.clean(text)
        run = _Run(policy, cleaned)
        run.add("control_chars", "mask", removed)
        if len(cleaned) > policy.max_input_chars:
            run.add("input_too_long", "block")
        if detectors.has_role_spoof(cleaned):
            run.add("input_role_spoof", "block")
        run.apply(
            (detectors.find_secrets(cleaned), policy.input_secrets),
            (detectors.find_pii(cleaned, policy.locales), policy.input_pii),
        )
        return run.decision(text)

    async def check_retrieved(self, text: str, *, tenant_id: str) -> GuardrailDecision:
        policy = self._policies.for_tenant(tenant_id)
        cleaned, removed = detectors.clean(text)
        run = _Run(policy, cleaned)
        run.add("control_chars", "mask", removed)
        run.apply(
            (detectors.find_injection(cleaned), policy.retrieved_injection),
            (detectors.find_secrets(cleaned), policy.retrieved_secrets),
            (detectors.find_pii(cleaned, policy.locales), policy.retrieved_pii),
        )
        if policy.retrieved_external_images != "allow":
            stripped, n = detectors.strip_external_urls(run.text, images_only=True)
            if policy.retrieved_external_images == "mask":
                run.text = stripped
            run.add("external_url", policy.retrieved_external_images, n)
        return run.decision(text)

    async def check_output(
        self, text: str, *, tenant_id: str, allowed_urls: frozenset[str] = frozenset()
    ) -> GuardrailDecision:
        policy = self._policies.for_tenant(tenant_id)
        cleaned, removed = detectors.clean(text)
        run = _Run(policy, cleaned)
        run.add("control_chars", "mask", removed)
        run.apply((detectors.find_secrets(cleaned), policy.output_secrets))
        if any(marker in normalize(cleaned) for marker in policy.leak_markers):
            run.add("prompt_leak", policy.output_leak)
        if policy.output_external_urls != "allow":
            stripped, n = detectors.strip_external_urls(run.text, allowed_urls)
            if policy.output_external_urls == "mask":
                run.text = stripped
            run.add("external_url", policy.output_external_urls, n)
        # PII last, on the rewritten text, so the offsets are valid
        run.apply((detectors.find_pii(run.text, policy.locales), policy.output_pii))
        return run.decision(text)
