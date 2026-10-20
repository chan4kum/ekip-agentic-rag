"""Decorators that put the guardrail engine around the ports the agent already uses, so the
integrator wraps the model and the retriever without touching agent code.

Both fail CLOSED: an engine exception counts as a block. Neither ever puts blocked text, matched
text or an exception message into an error, a log or an audit event.
"""

import html
import re
from collections.abc import Awaitable, Sequence

from ekip_contracts.audit import AuditEvent, AuditLogger
from ekip_contracts.guardrails import Finding, GuardrailDecision, GuardrailEngine
from ekip_contracts.llm import ChatMessage, ChatModel, LLMError
from ekip_contracts.retrieval import RetrievedChunk, Retriever, SearchScope
from ekip_observability.logging import get_logger

from ekip_guardrails.audit import NullAuditLogger
from ekip_guardrails.scope import GuardScope, current_scope

log = get_logger("ekip.guardrails")

INPUT_BLOCKED = "guardrail: input blocked"  # LLMError messages are fixed phrases
OUTPUT_BLOCKED = "guardrail: output blocked"
_QUESTION = re.compile(r"(<question>)(.*)(</question>)", re.DOTALL)


def _engine_failure(policy_version: str = "") -> GuardrailDecision:
    return GuardrailDecision("block", "", (Finding("engine_error"),), policy_version)


def _outcome(decision: GuardrailDecision) -> str:
    return {"allow": "allowed", "mask": "masked", "block": "blocked"}[decision.action]


class GuardedChatModel:
    """ChatModel decorator. Checks the user's question on the way in and the reply on the way out.

    Retrieved passages are NOT re-checked here: GuardedRetriever neutralises them before they are
    ever placed in a prompt, so this wrapper stays cheap and does not mangle documents.
    """

    def __init__(
        self, inner: ChatModel, engine: GuardrailEngine, audit: AuditLogger | None = None
    ) -> None:
        self._inner, self._engine = inner, engine
        self._audit: AuditLogger = audit or NullAuditLogger()

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int = 800, temperature: float = 0.0
    ) -> str:
        scope = current_scope() or GuardScope(tenant_id="")
        guarded = await self._guard_input(list(messages), scope)
        reply = await self._inner.complete(guarded, max_tokens=max_tokens, temperature=temperature)
        decision = await self._check(
            self._engine.check_output(
                reply, tenant_id=scope.tenant_id, allowed_urls=frozenset(scope.allowed_urls)
            )
        )
        self._record("guardrail.output", decision, scope)
        if decision.blocked:
            raise LLMError(OUTPUT_BLOCKED)
        return decision.text

    async def _guard_input(
        self, messages: list[ChatMessage], scope: GuardScope
    ) -> list[ChatMessage]:
        index = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), -1
        )
        if index < 0:
            return messages
        content = messages[index].content
        match = _QUESTION.search(content)
        # the prompt builder HTML-escapes the question; undo that so the engine sees the real text
        text = html.unescape(match.group(2)) if match else content
        decision = await self._check(self._engine.check_input(text, tenant_id=scope.tenant_id))
        self._record("guardrail.input", decision, scope)
        if decision.blocked:
            raise LLMError(INPUT_BLOCKED)
        if decision.action == "mask":
            if match:
                safe = html.escape(decision.text, quote=False)
                content = content[: match.start(2)] + safe + content[match.end(2) :]
            else:
                content = decision.text
            messages[index] = ChatMessage(messages[index].role, content)
        return messages

    @staticmethod
    async def _check(pending: Awaitable[GuardrailDecision]) -> GuardrailDecision:
        try:
            return await pending
        except Exception as exc:  # fail closed on ANY engine failure
            log.warning("guardrail_engine_failed", error_type=type(exc).__name__)
            return _engine_failure()

    def _record(self, action: str, decision: GuardrailDecision, scope: GuardScope) -> None:
        if decision.action == "allow":
            return  # only denials and rewrites are audit-worthy
        self._audit.record(
            AuditEvent(
                action=action,
                outcome=_outcome(decision),  # type: ignore[arg-type]
                tenant_id=scope.tenant_id,
                subject=scope.subject,
                request_id=scope.request_id,
                codes=decision.codes,
                policy_version=decision.policy_version,
            )
        )


class GuardedRetriever:
    """Retriever decorator. Neutralises instruction-like text in passages (or drops the passage),
    and records the legitimate source URLs for the output exfiltration check.

    It adds no authorization: ACL and tenant filtering stay INSIDE the wrapped retriever's query
    (I-4, I-5). This only treats what came back as untrusted data (I-3).
    """

    def __init__(
        self, inner: Retriever, engine: GuardrailEngine, audit: AuditLogger | None = None
    ) -> None:
        self._inner, self._engine = inner, engine
        self._audit: AuditLogger = audit or NullAuditLogger()

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        hits = await self._inner.search(query, scope=scope, limit=limit)
        guard = current_scope()
        kept: list[RetrievedChunk] = []
        for hit in hits:
            try:
                decision = await self._engine.check_retrieved(hit.text, tenant_id=scope.tenant_id)
            except Exception as exc:  # fail closed: an unchecked passage is not shown
                log.warning("guardrail_engine_failed", error_type=type(exc).__name__)
                decision = _engine_failure()
            if decision.action != "allow":
                self._audit.record(
                    AuditEvent(
                        action="guardrail.retrieved",
                        outcome=_outcome(decision),  # type: ignore[arg-type]
                        tenant_id=scope.tenant_id,
                        subject=guard.subject if guard else "",
                        resource=hit.chunk_id,
                        request_id=guard.request_id if guard else "",
                        codes=decision.codes,
                        policy_version=decision.policy_version,
                    )
                )
            if decision.blocked:
                continue
            if guard is not None and hit.source_uri:
                guard.allowed_urls.add(hit.source_uri)
            kept.append(
                hit
                if decision.action == "allow"
                else hit.model_copy(update={"text": decision.text})
            )
        return kept
