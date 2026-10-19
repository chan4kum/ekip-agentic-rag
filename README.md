# EKIP: Enterprise Knowledge Intelligence Platform

A multi-tenant **agentic RAG platform** designed for 1M+ documents with continuous
incremental ingestion, document-level access control, and governed tool use through an
**MCP tool gateway**. Built step by step as a production-style reference project, with
architecture decision records and tests at every stage.

> **Status: in active development.** See the roadmap below for what is built today.

## Why this project
Most RAG demos stop at "chat with a PDF". EKIP focuses on the parts that decide whether an
AI system survives a design review: identity and authorization, tenant isolation, safe
incremental ingestion, measurable retrieval quality, bounded agents, secure tool access,
and observability.

## Architecture (target)
```
Client -> API Gateway -> FastAPI (OIDC/JWT, RBAC/ABAC, rate limits)
                              |
                      LangGraph agents (router, planner, retrieval, analysis, verifier)
                       |            |                 |
                 Retrieval      LLM gateway      MCP Tool Gateway -> internal / external APIs
              (BM25 + vector,   (provider-         (allowlists, authz, audit, approvals)
               ACL, rerank)      agnostic)
                       |
        OpenSearch · PostgreSQL · Redis · Object storage

Ingestion (independent): connectors -> Kafka -> Temporal -> parse -> chunk -> embed -> index
Cross-cutting: OpenTelemetry · Langfuse · Prometheus/Grafana · Terraform · Kubernetes · GitOps
```

## Engineering principles
- **Modular monorepo** with enforced import boundaries ([ADR-001](docs/adr/001-modular-monorepo.md))
- **SOLID + dependency injection**: ports and adapters, everything testable with fakes ([ADR-006](docs/adr/006-solid-di.md))
- **Security invariants** that every feature is tested against ([docs/security-invariants.md](docs/security-invariants.md))
- **Provider-agnostic LLM layer** (OpenAI-compatible endpoints; Groq by default)

## Roadmap
| Phase | Scope | Status |
|---|---|---|
| 0 | Architecture decisions, security invariants | Done |
| 1 | Foundation: toolchain, FastAPI, config, structured logging, tracing, hardened Docker Compose (reverse proxy, egress proxy), CI | Done |
| 2 | Identity and backend: OIDC/JWT validation (Keycloak), tenant isolation, RBAC, Redis rate limits, RFC 9457 errors, deadlines, idempotency, SSE chat, security regression suite | Done |
| 3 | Single-document ingestion: hardened parsers (text/HTML/PDF/DOCX, images), structure-aware chunking, Gemini embeddings with cache, OpenSearch hybrid index with ACL fields, resumable `IngestDocument` pipeline | Done |
| 4 | Incremental ingestion: change events, state-based out-of-order handling, Redpanda/Kafka consumer with lanes and DLQ, Temporal single-flight workflow, tenant-scoped embedding reuse, deletes, ACL changes, source reconciliation | Done |
| 5 | Retrieval: hybrid search, reranking, citations, evaluation harness | Planned |
| 6 | Single-agent RAG with LangGraph and checkpoints | Planned |
| 7 | Multi-step agents: planner, verifier, budgets, human approval | Planned |
| 7A | **MCP tool gateway**: allowlists, OAuth, token exchange, resilience, audit | Planned |
| 8+ | Caching, guardrails, LLMOps, observability, scale validation | Planned |

## Quick start
Requires [uv](https://docs.astral.sh/uv/).
```bash
cp .env.example .env     # then fill in your own keys; .env is git-ignored
uv sync
uv run pytest
docker compose up -d --build --wait && bash scripts/smoke.sh   # live stack + smoke test
```
macOS note: if Python cannot import the workspace packages after `uv sync`, run `chflags -R nohidden .venv` (the OS `hidden` flag makes Python skip editable `.pth` files).

## API surface (Phase 2; no new routes in Phase 3)
| Endpoint | Auth | Notes |
|---|---|---|
| `GET /health`, `GET /ready` | public | liveness / dependency readiness |
| `GET /api/v1/search` | Bearer JWT, role with `documents:search` | tenant comes only from the validated token; per-user and per-tenant rate limits |
| `POST /api/v1/chat` | Bearer JWT | Server-Sent Events: `meta`, `token`, `citations`, `done`, `error`; own deadline and heartbeats |

Security posture: RS256-only allowlist, JWKS cache with rotation, generic 401s, fail-closed authorization and rate limiting, problem+json errors. A route-inventory test fails any new route until it is classified public or protected, and every protected route is attacked with forged, expired, mis-audienced and tampered tokens (`tests/security`).

## Ingestion pipeline (Phase 3)
`submit` (raw bytes to the object store, version registered) then `run`: **parse → chunk → embed → index → INDEXED**, each stage persisted before the status advances.
- **Parsers treat input as hostile:** size/page/zip-expansion limits, magic-byte checks, hidden HTML text dropped, no remote image fetches, generic parse errors.
- **Chunking is deterministic (no LLM calls):** section-bounded, table/code aware, boilerplate stripped, stable chunk ids; images become searchable through an image chunk (caption + alt + OCR/description) with the bytes in the object store.
- **Fewer model calls:** identical texts embedded once per call, batches of 100, cache keyed by (model version, task, text hash), so unchanged chunks and retries cost nothing.
- **One index per embedding model version** (HNSW, cosine, Lucene engine) with tenant and ACL filters applied *during* vector search; an empty ACL matches nobody.
- **Recoverable:** an embedding or index outage leaves the document `FAILED` with a fixed reason code and a `retryable` flag; the retry resumes without re-parsing or re-embedding. An older version can never overwrite a newer one.

## Incremental ingestion (Phase 4)
`SourceChange` events (upsert · acl_change · delete) → **Kafka-API topics** (live and backfill lanes, DLQ) → `ChangeHandler` → Temporal workflow → `IngestDocument`.
- **Decides from state, not arrival order:** separate source-time clocks for content, ACL and tombstones; duplicates, replays and stale snapshots are harmless; a delete beats an equal-time update.
- **At-least-once with safe redelivery:** offsets commit only after the batch is handled or parked in the DLQ; per-document ordering via message keys; the live lane is always drained before backfill, and deletes/permission changes never queue behind a replay.
- **Poison messages never block the stream:** retryable failures back off with jitter then park; DLQ keeps the original bytes plus a reason *code* (never exception text).
- **Durable processing:** one Temporal workflow per document version (`USE_EXISTING`, so concurrent triggers run it once); `retryable` drives Temporal's retry policy; failures cross the boundary as codes.
- **Cheap updates:** a tenant-scoped Redis embedding cache means editing one paragraph of a ten-section document re-embeds exactly one chunk.
- **Reconciliation:** periodic source inventory diff repairs missed creates, edits, permission changes and deletes by publishing ordinary events; incomplete listings abort before any delete, and a mass-delete guard withholds suspicious deletions.

## Repository layout
```
apps/        deployable workloads (api, agent-runtime, retrieval-service, ingestion-worker, mcp-servers)
packages/    shared libraries (contracts, identity, authorization, llm-gateway, observability, ...)
tests/       unit · integration · contract · security · e2e
docs/        architecture decision records, security invariants, learning log
```

## Documentation
- [Architecture decision records](docs/adr/)
- [Security invariants](docs/security-invariants.md)
- [Learning log](docs/learning-log.md)
