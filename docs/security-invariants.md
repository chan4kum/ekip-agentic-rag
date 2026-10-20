# Security invariants (non-negotiable)

Status: Accepted · Date: 2026-10-19

- **I-1 Fail closed.** If identity, tenant, or permissions can't be established, deny. Never "allow on error".
- **I-2 The LLM is not a security boundary.** Authorization is decided by code (JWT validation + Authorizer), never by a prompt or model output.
- **I-3 Untrusted data.** Retrieved text, tool output and uploaded documents are evidence, never instructions, and can't expand which tools an agent may call.
- **I-4 Tenant isolation** is enforced server-side on every read, write, cache key and tool call. `tenant_id` comes only from a validated token, never from the request body or model output.
- **I-5 Authorize before the LLM.** ACL filtering happens inside the search query, not after generation.
- **I-6 Every tool call is checked server-side:** caller identity, tool allowlist, validated arguments, timeout, audit record. Write tools also require explicit human approval.
- **I-7 No token passthrough.** Downstream services receive a delegated/exchanged credential scoped to them, not the user's original JWT.
- **I-8 Secrets** live only in environment / secret manager. Never in code, prompts, logs, traces, or git. `.env` is git-ignored.
- **I-9 Caches never bypass authorization.** Keys include tenant and permission-scope version; revocation invalidates entries.
- **I-10 Bounded agents.** Max steps, tool calls, tokens and wall-clock time on every run.
- **I-11 Audit.** Sensitive actions emit an event: who, what, tenant, tool, args hash, outcome, trace id.

## Test mapping
| Invariant | First test |
|---|---|
| I-1, I-4 | Phase 2.8 cross-tenant / expired-token tests |
| I-3, I-6, I-7 | Phase 7A.14 injection, unauthorized-tool, passthrough tests |
| I-9 | Phase 8 cache-leakage tests |

## Phase 9 additions (guardrails and hardening)

Guardrails are layered defence on top of the invariants above, never a replacement (see
`docs/adr/011-guardrails.md`).

- **I-12 Retrieved text is neutralised before a prompt.** Instruction-like sentences, credential
  shapes and external auto-loading images are removed from passages; facts are kept. This raises
  attacker cost; it is NOT a boundary (I-2, I-3).
- **I-13 No exfiltration channel in replies.** A model reply may contain external URLs only if
  they are the citation sources of that request; all other external images, links and bare URLs
  are removed before anyone sees the reply.
- **I-14 Replies are screened whole.** Credentials and verbatim system-prompt fragments block the
  reply (the user sees the fixed failure message); PII is masked per tenant policy.
- **I-15 Questions are screened.** Oversized input, control/bidi/zero-width characters and
  role-spoof markers (`system:`, chat-template tokens, forged `<source>`/`<question>` elements)
  are blocked or stripped; PII and credentials are masked before any model sees them.
- **I-16 Guardrails fail closed and leak nothing.** An engine failure is a block. Decisions,
  errors, logs and audit events carry fixed finding codes and counts, never matched text; audit
  strings are limited to a short safe alphabet.
- **I-17 Policy is versioned data per tenant;** unknown policy keys are rejected, and every
  decision records the policy version that produced it.

| Invariant | Tests |
|---|---|
| I-3, I-12 | `test_guardrails_agent_attacks.py::test_instructions_inside_a_document_never_reach_the_model`, `::test_a_document_cannot_smuggle_prompt_markup_to_forge_a_source`; `test_injection_corpus.py` (payloads, obfuscations, no false positives) |
| I-2, I-3 | `test_injection_corpus.py::test_a_paraphrased_injection_that_slips_past_the_screen_still_cannot_widen_anything` |
| I-4, I-5 | `test_guardrails_agent_attacks.py::test_another_tenants_document_never_reaches_the_model_or_the_answer`, `::test_acl_restricted_documents_are_invisible_to_unlisted_callers`, `::test_text_in_the_question_cannot_widen_the_search_scope`, `::test_another_users_history_is_never_loaded` |
| I-13 | `::test_a_document_asking_the_model_to_exfiltrate_via_an_image_is_stripped`, `::test_a_model_that_obeys_and_links_to_an_attacker_site_is_defanged`, `::test_a_link_to_the_citation_source_itself_is_kept`; `test_guardrails_detectors.py` URL tests |
| I-14 | `::test_a_secret_in_the_reply_is_never_shown_and_the_user_gets_a_fixed_message`, `::test_pii_in_the_reply_is_masked_but_the_answer_still_cites`, `::test_a_reply_that_reproduces_the_system_prompt_is_blocked`, `::test_a_user_asking_for_the_system_prompt_gets_no_prompt_text` |
| I-15 | `::test_an_oversized_question_is_stopped_before_the_model`, `::test_role_spoofing_and_unicode_tricks_never_reach_the_model`, `::test_invisible_characters_are_stripped_from_the_question_the_model_sees`, `::test_card_numbers_in_a_question_are_masked_before_the_model_and_the_search`, `::test_a_poisoned_earlier_turn_is_data_and_cannot_force_a_leak` |
| I-16 | `test_guardrails_guarded.py::test_engine_failure_fails_closed_on_input_and_output`, `::test_retriever_drops_a_passage_it_cannot_check`, `::test_audit_events_never_contain_content`; `test_guardrails_audit.py`; `::assert_audit_is_clean` in the attack suite; `test_guardrails_nemo.py::test_an_unreachable_rail_fails_closed` |
| I-17 | `test_guardrails_engine.py::test_each_tenant_can_have_its_own_policy`, `::test_policy_from_mapping_rejects_unknown_keys_and_bad_actions` |
| I-11 | `test_guardrails_guarded.py` (blocked/masked events with codes, tenant, subject, request id) |

Not covered by Phase 9 guardrails (tracked elsewhere): tool authorization, argument validation and
SSRF/SQL-injection of tools belong to the Phase 7A gateway tests; cache leakage to Phase 8.
