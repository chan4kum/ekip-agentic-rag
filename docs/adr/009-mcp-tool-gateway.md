# ADR-009: MCP tool gateway (Phase 7A)

Status: accepted, October 2026

## Context

Agents need to act: search documents, read approved business data, search the web, open a ticket.
Letting an LLM call APIs directly puts the security boundary inside the model, which we reject
(invariants I-1, I-2). We also want reusable, discoverable tools that other agents can share
(MCP), without making MCP responsible for our authorization.

## Decision

All tool use goes through one `ToolGateway` (`packages/tools`). Agents depend only on the
`ToolInvoker` port in `ekip_contracts.tools`, so the gateway, the transport and every backend are
replaceable.

### What we verified before building (October 2026)

- The MCP spec dated 2026-07-28 (modelcontextprotocol.io/specification/2026-07-28) is **stateless**:
  requests are self-contained and capabilities are negotiated per request. There is no session to
  keep alive, so the backend opens a short-lived client per call and nothing is pooled yet.
  Spec text also says tool descriptions are untrusted unless the server is trusted, and hosts must
  get consent before invoking tools: the registry (our own reviewed schemas) and the approval flow
  implement that.
- The installed Python SDK is `mcp` 2.3.0: `FastMCP` is now `MCPServer`
  (`mcp.server.mcpserver`), `Client` takes an `MCPServer` for in-process tests, and `_meta` is
  a TypedDict with extra items, which we use for the trace context and the delegation token.
  `CallToolResult` has `is_error`; the server handler reads `ctx.request_context.meta`.
- `Client(...)` can auto-negotiate (discover, falling back to the handshake) so older servers keep
  working during the transition.

### Pipeline (every call, this order)

registry lookup -> authorization -> argument validation -> rate limit (user + tenant) ->
approval (write tools) -> circuit breaker -> bulkhead -> backend (timeout, retry only if
idempotent) -> output cap + untrusted wrapping -> audit + OpenTelemetry span.

- **Fail closed** everywhere: an authorizer error, a rate limiter outage or a missing backend is a
  refusal (backends are checked at startup).
- **Authorization before validation**: a caller without access learns nothing about schemas.
- **Registry** (`tools/registry.yaml`) is reviewed config: name, risk, roles, JSON-schema subset,
  timeouts, size caps, retry policy. The schema subset (ekip_tools.schema) forces bounded strings
  and numbers and rejects unknown fields. Error text names fields, never values.
- **Identity downstream**: the user's access token is never forwarded. For MCP servers the gateway
  mints a signed delegation token (HMAC, audience = the tool, 30 s TTL, tenant and subject inside)
  and sends it in `_meta`. The server verifies it and re-authorizes (defence in depth). For HTTP
  APIs a `CredentialProvider` does an OAuth token exchange (RFC 8693). Production should swap
  the HMAC signer for the identity provider's token exchange; the contract is the same.
- **Write tools** (`create_ticket`) need a human approval bound to the exact arguments (sha256
  digest), the requester, tenant and tool; single use; short TTL; the approver must hold an
  approver role, be in the same tenant and (by default) be a different person. The gateway returns
  `APPROVAL_REQUIRED` with an `ApprovalRequest`; Lane A's graph checkpoints it and resumes with
  the grant. Writes are never retried.
- **Output is data**: sanitized (control/format characters removed), capped, wrapped in
  `<tool_output tool="...">` with the marker escaped inside, so a result cannot forge the wrapper.
  This marks injected text; it does not neutralize it. The real defences are the allowlist,
  re-authorization, and approvals, tested in `tests/security/test_tools_attacks.py`.
- **Resilience** per tool: circuit breaker (5 failures, 30 s, single half-open probe), bulkhead
  (reject, never queue), retry with jitter for idempotent tools only.
- **SSRF**: https only, no credentials, exact-host allowlist, port 443, no IP literals, resolved
  addresses must all be public, redirects never followed, responses capped at 256 KiB.
- **SQL**: no free-form SQL. `query_database` selects an allowlisted, tenant-scoped view; the
  statement text is built from validated identifiers and all values are bound parameters
  (`$1` tenant from the context, `$2` key, `$3` limit).
- **Action agent** (`ekip_agents.action`): bounded loop (steps, tool calls, repeat guard, refusal
  budget) over the `ToolInvoker` port; stops with `NEEDS_APPROVAL` for writes and resumes from a
  plain-data `PendingAction`.

### MCP vs direct calls

`search_documents` is an MCP server (`apps/mcp-servers/docs`): reusable and independently
deployable. `create_ticket`, `web_search` and `query_database` are direct typed backends: simple,
latency-sensitive, no extra hop. Both kinds sit behind the same gateway, so policy is uniform.

## Consequences

- One place to audit, rate limit and trace tool use (`ekip.tool.call` span, `gen_ai.*` attributes;
  no arguments or outputs in spans or audit lines, only a digest).
- Adding a tool is: a registry entry (reviewed), a backend, tests. Nothing in the agent changes.
- Costs: an extra hop for MCP tools (a new client per call until a pooled Streamable HTTP client is
  justified by measurements) and in-memory rate-limit/replay stores in tests.

## Known gaps / follow-ups

- Production wiring: Redis `ReplayGuard`, Redis rate limiters, token-exchange `CredentialProvider`,
  asyncpg `ViewExecutor` on a read-only role, Streamable HTTP + OAuth for the docs server,
  Kubernetes egress NetworkPolicy to close the DNS-rebinding gap (user-owned Kubernetes stage).
- The audit sink failing is logged but does not block; for write tools consider writing the
  intent record before executing.
- `web_search` sends the user's text to an external provider: wire `query_filter` to the PII
  guardrail (Lane D) before enabling it for real data.
- Roles are today's `ekip-user` / `ekip-admin`; `query_database` requires `ekip-admin` until an
  analyst role exists. Approvers are `ekip-admin`.
- `tools/registry.yaml` is parsed with PyYAML, which is a workspace dev dependency but not yet a
  declared dependency of `ekip-tools`.
