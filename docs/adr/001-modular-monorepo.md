# ADR-001: Modular monorepo with separately deployable workloads

Status: Accepted · Date: 2026-10-19

## Context
One engineer, 1M+ documents. API, agent runtime, retrieval and ingestion
must scale independently, but the right service boundaries are not yet
known. Splitting into many microservices now would add network hops,
deploy overhead and debugging pain.

## Decision
One git repo, one uv workspace.
- `apps/*`      deployable workloads: api, agent-runtime, retrieval-service,
                ingestion-worker, mcp servers
- `packages/*`  shared libraries: contracts, identity, authorization,
                llm-gateway, observability, caching, guardrails, tools

Rules:
1. apps may import packages. packages never import apps.
2. apps never import other apps; they talk via contracts
   (HTTP / events / MCP).
3. Domain logic never imports infra SDKs (OpenSearch, Redis, OpenAI);
   it depends on ports (Protocols). Adapters live at the edge.
4. Rules are enforced in CI with import-linter.

## Consequences
+ One CI, atomic cross-cutting changes, easy to extract a service later
- Needs tooling (import-linter) to stop boundary drift
- Shared lockfile: one dependency upgrade affects every app

## Rejected alternatives
- Polyrepo: too much overhead for one engineer
- One microservice per agent: latency, cost, tracing complexity
