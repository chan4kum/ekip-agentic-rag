# ADR-011: Layered guardrails behind a port, never as the security boundary

Status: accepted (2026-10-20). Applies to Phase 9.

## Context
Prompt-injection, PII exposure, credential leaks, system-prompt extraction and data exfiltration
through rendered links are the main content risks of an agentic RAG system. A classifier or a
sentence in the prompt can be bypassed, so guardrails must reduce risk without being the thing
that grants or denies access.

## Decision
1. **Authorization stays elsewhere (I-2, I-4, I-5).** Tenant and ACL filtering happen inside the
   search query; the guardrails never decide who may read a document. They treat what comes back as
   untrusted data (I-3) and screen what goes out.
2. **A `GuardrailEngine` port** (`ekip_contracts.guardrails`) with three checks at the three trust
   boundaries: `check_input` (the question), `check_retrieved` (each passage) and `check_output`
   (the model reply). A decision is `allow`, `mask` or `block`, with fixed finding codes and counts
   only - never the matched text - and the policy version that produced it. `decision.text` is
   already safe to use (original, redacted or empty), so forgetting to branch cannot leak.
3. **A deterministic default engine** (`PolicyGuardrailEngine`): input length/control characters/
   role-spoof markers; PII (email, phone, card with Luhn, IBAN with mod-97, US SSN, India Aadhaar
   with Verhoeff and PAN, selected by a locale list); credentials; instruction-like sentences in
   retrieved text (neutralised, facts kept); system-prompt leakage (configured markers and
   `prompt_shingles` of the real prompt); and an **exfiltration check**: any external image, link
   or URL in a reply that is not one of the citation sources is removed. Text is stripped of
   invisible characters and NFKC-normalized before matching, so zero-width, bidi and fullwidth
   tricks do not hide a pattern.
4. **Policy is data, versioned and per tenant** (`GuardrailPolicy`, `PolicySet`). Unknown keys are
   rejected so a typo cannot silently weaken a tenant. Defaults: input/output PII masked, reply
   secrets and prompt leaks blocked, retrieved PII allowed (the caller may read the document).
5. **Integration by decoration, not by editing agents.** `GuardedChatModel` wraps the `ChatModel`
   port (screens the `<question>` on the way in and the reply on the way out); `GuardedRetriever`
   wraps the `Retriever` port (neutralises passages, records legitimate source URLs). A per-request
   `GuardScope` (a ContextVar holding one mutable object) carries tenant, subject, request id and
   the allowed-URL set between them. Both fail closed: an engine exception is a block, surfaced to
   the agent as a fixed `LLMError` (so the user sees the existing generic failure message) or as a
   dropped passage.
6. **Audit (I-11)** through an `AuditLogger` port. Events hold who/tenant/action/resource/outcome/
   codes/policy version and nothing else; every string is restricted to a short safe alphabet.
   Allowed checks are not logged; masks, blocks and denials are.
7. **Optional NeMo Guardrails** adapter behind the same port, lazily imported and NOT a dependency.
   It can only make a deterministic decision stricter; an unreachable rail is a block.

## Consequences
- Every control has a unit test and an attack test against the real agent graph
  (`tests/security/test_guardrails_agent_attacks.py`, `test_injection_corpus.py`).
- The injection screen is a heuristic: it raises attacker cost and removes obvious payloads, and a
  dedicated test proves that when it misses a paraphrase the real boundaries still hold.
- Masking is regex-based: it will miss exotic formats and can over-match long digit strings
  (a 10-15 digit number looks like a phone number). Tenants can switch detectors per policy.
- Streaming token-by-token output is not covered: the reply is checked whole, which is how the
  agent produces it today. A streaming design would need windowed output checks first.

## Integration (done by the application wiring, not by this package)
```python
engine = PolicyGuardrailEngine(PolicySet(...))          # prompt_shingles(SYSTEM_PROMPT) in leak_markers
model = GuardedChatModel(real_model, engine, StructlogAuditLogger())
retriever = GuardedRetriever(real_retriever, engine, audit)
with guard_scope(ctx.tenant_id, subject=ctx.subject, request_id=request_id):
    await run_agent(graph, question, ctx, conversation_id)
```
