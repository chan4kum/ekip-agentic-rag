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
