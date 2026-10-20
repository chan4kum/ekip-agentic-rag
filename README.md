# EKIP: Enterprise Knowledge Intelligence Platform

[![CI](https://github.com/chan4kum/ekip-agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/chan4kum/ekip-agentic-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Status](https://img.shields.io/badge/status-in%20active%20development-orange)

A multi-tenant **agentic RAG platform** designed for 1M+ documents with continuous incremental
ingestion, document-level access control and governed tool use. It is built step by step as a
production-style reference project: every stage ships with architecture decisions, failure-mode
tests and a written account of what was learned.

## Why this project

Most RAG demos stop at "chat with a PDF". EKIP concentrates on the parts that decide whether an AI
system survives a design review:

- **Identity and authorization:** OIDC/JWT validation, tenant isolation, RBAC, document-level ACLs enforced *inside* the search queries.
- **Safe incremental ingestion:** out-of-order events, replays, deletes and permission changes handled from state, not arrival order.
- **Measurable retrieval quality:** an evaluation harness with a hard gate on access leaks.
- **Bounded, verifiable agents:** typed state, step budgets, citation checks, fail-closed behaviour.
- **Observability and operations:** structured logs, tracing, rate limits, hardened containers.

## Architecture

```
Client -> Reverse proxy -> FastAPI (OIDC/JWT, RBAC, rate limits, SSE)
                               |
                       LangGraph agent
         load_history -> classify -> rewrite -> retrieve -> answer -> verify -> save_turn
               |                         |                   |
        Conversation store        Hybrid retrieval      Chat model port
        (memory | Postgres)    (BM25 + kNN, ACL, RRF)  (OpenAI-compatible)
                                         |
              OpenSearch · PostgreSQL · Redis · Object storage

Ingestion (independent): connectors -> Redpanda/Kafka -> Temporal -> parse -> chunk -> embed -> index
Planned: MCP tool gateway · guardrails · OpenTelemetry/Langfuse · Kubernetes · GitOps
```

The API, agent runtime, retrieval and ingestion workers scale independently. Dependencies point
inward through ports (`typing.Protocol`); an import-linter contract set (12 contracts) enforces the
boundaries in CI.

## Technology

| Concern | Choice |
|---|---|
| API | Python 3.12, FastAPI, Server-Sent Events |
| Identity | Keycloak (OIDC), RS256-only JWT validation with JWKS rotation |
| Agent orchestration | LangGraph |
| Retrieval | OpenSearch (BM25 + HNSW kNN), Reciprocal Rank Fusion |
| Embeddings | Google Gemini (`gemini-embedding-001`, 768 dims) behind a port, with a tenant-scoped cache |
| Chat model | Any OpenAI-compatible endpoint (OpenAI, Groq, DeepSeek, vLLM) |
| Ingestion | Redpanda (Kafka API), Temporal, structure-aware chunking |
| State | PostgreSQL (metadata, conversations), Redis (rate limits, caches, idempotency) |
| Quality | pytest, mypy (strict), ruff, import-linter, GitHub Actions |

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Architecture decisions, security invariants | Done |
| 1 | Foundation: toolchain, FastAPI, config, structured logging, tracing, hardened Docker Compose, CI | Done |
| 2 | Identity and backend: OIDC/JWT, tenant isolation, RBAC, rate limits, RFC 9457 errors, deadlines, idempotency, SSE chat, security regression suite | Done |
| 3 | Single-document ingestion: hardened parsers, structure-aware chunking, embeddings with cache, OpenSearch hybrid index with ACL fields | Done |
| 4 | Incremental ingestion: change events, out-of-order handling, Kafka consumer with lanes and DLQ, Temporal workflows, deletes, ACL changes, reconciliation | Done |
| 5 | Retrieval: ACL-pushdown hybrid search, optional reranker, cited context assembly, evaluation harness with an access-leak gate | Done |
| 6 | Single-agent RAG: LangGraph agent, conversation memory, query rewriting, OpenAI-compatible model, `/api/v1/chat` wiring | Done |
| 7 | Multi-step agents: planner, groundedness verifier, budgets, human approval, durable checkpoints | Planned |
| 7A | **MCP tool gateway**: allowlists, OAuth, token exchange, resilience, audit | Planned |
| 8+ | Caching, guardrails, LLMOps, observability, scale validation, Kubernetes | Planned |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env      # fill in your own keys; .env is git-ignored and never printed
uv sync
uv run pytest             # unit, contract and security suites run without any service
docker compose up -d --build --wait && bash scripts/smoke.sh   # live stack + smoke test
```

To answer chat with the real agent instead of the placeholder, set in `.env`:

```
CHAT_BACKEND=agent
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=<model name>
LLM_API_KEY=<key>
CONVERSATION_STORE=postgres      # or memory
```

macOS note: if Python cannot import the workspace packages after `uv sync`, run
`chflags -R nohidden .venv` (the OS `hidden` flag makes Python skip editable `.pth` files).

## API

| Endpoint | Auth | Notes |
|---|---|---|
| `GET /health`, `GET /ready` | public | liveness and dependency readiness |
| `GET /api/v1/search` | Bearer JWT, `documents:search` | tenant comes only from the validated token; per-user and per-tenant rate limits |
| `POST /api/v1/chat` | Bearer JWT | SSE events: `meta`, `token`, `citations`, `status`, `done`, `error`; optional `conversation_id` continues a thread |

**Security posture:** RS256-only allowlist, JWKS cache with rotation, generic 401s, fail-closed
authorization and rate limiting, problem+json errors. A route-inventory test fails any new route
until it is classified public or protected, and every protected route is attacked with forged,
expired, mis-audienced and tampered tokens (`tests/security`).

## Design notes by phase

<details>
<summary><strong>Ingestion pipeline (Phase 3)</strong></summary>

`submit` (raw bytes to the object store, version registered) then `run`: **parse, chunk, embed, index**, each stage persisted before the status advances.
- **Parsers treat input as hostile:** size, page and zip-expansion limits, magic-byte checks, hidden HTML text dropped, no remote image fetches, generic parse errors.
- **Deterministic chunking (no LLM calls):** section-bounded, table and code aware, boilerplate stripped, stable chunk ids; images become searchable through an image chunk.
- **Fewer model calls:** identical texts embedded once, batches of 100, cache keyed by (model version, task, text hash).
- **One index per embedding model version** with tenant and ACL filters applied *during* vector search; an empty ACL matches nobody.
- **Recoverable:** an outage leaves the document `FAILED` with a reason code and a `retryable` flag; retry resumes without re-parsing or re-embedding. An older version can never overwrite a newer one.
</details>

<details>
<summary><strong>Incremental ingestion (Phase 4)</strong></summary>

`SourceChange` events (upsert, acl_change, delete) flow through Kafka-API topics (live and backfill lanes, DLQ) to a `ChangeHandler` and a Temporal workflow.
- **Decides from state, not arrival order:** separate source-time clocks for content, ACL and tombstones; duplicates, replays and stale snapshots are harmless; a delete beats an equal-time update.
- **At-least-once with safe redelivery:** offsets commit only after a batch is handled or parked in the DLQ; the live lane drains before backfill, and deletes and permission changes never queue behind a replay.
- **Poison messages never block the stream:** retryable failures back off with jitter and park; the DLQ keeps the original bytes plus a reason *code*, never exception text.
- **Cheap updates:** editing one paragraph of a ten-section document re-embeds exactly one chunk.
- **Reconciliation:** a periodic inventory diff repairs missed creates, edits, permission changes and deletes; incomplete listings abort before any delete, and a mass-delete guard withholds suspicious deletions.
</details>

<details>
<summary><strong>Retrieval (Phase 5)</strong></summary>

`query` plus verified identity runs **BM25 and kNN in parallel** (tenant and document ACL applied *inside* both queries), then RRF fusion, an optional reranker and a `ContextAssembler` that produces numbered, citable context.
- **Access control is in the query, not after it:** scope comes only from the verified token; a caller with no identity matches nothing.
- **Degrades without widening access:** if one search fails the other answers; a failing reranker keeps the fused order.
- **Context you can cite:** de-duplicated, capped per document, token-budgeted; retrieved text is escaped inside `<source id=n>` elements; `check_citations` flags markers the model invented.
- **Measured:** `scripts/eval_retrieval.py` compares BM25, vector, hybrid and hybrid+rerank (Recall@K, MRR, NDCG@K) and **fails if any variant returns a passage the caller may not read**. On the small synthetic set with real embeddings, vector search and hybrid both reached Recall 1.00, and the baseline reranker did not help, so none is enabled by default.
- **Honest limits:** 15 documents and 36 queries is a regression gate and a relative comparison, not a production quality estimate.
</details>

<details open>
<summary><strong>Single-agent RAG (Phase 6)</strong></summary>

A LangGraph agent over a typed `AgentState`: `load_history`, `classify`, `rewrite`, `retrieve`, `answer`, `verify`, `save_turn`, `finalize`.
- **Identity cannot be overwritten:** `question`, `security` and `conversation_id` are protected; a node that touches them, or invents a key, fails in every test. Retrieval scope is built only from the verified token, never from model output.
- **No evidence, no model call:** an empty context ends as `insufficient_evidence` with a fixed message instead of a guess.
- **Citations are verified before anything is shown:** every `[n]` must reference a source that was supplied and at least one must exist; a rejected draft is erased. This proves citation *validity*, not that the source supports the claim (groundedness is a Phase 7 verifier and evaluation task).
- **Conversation memory:** keyed by (tenant, user from the token, conversation id), so a guessed id returns nothing. Only answered turns are stored, never retrieved passages. Memory is best effort and can never fail an answer.
- **Follow-ups work:** "and the other bolt?" is rewritten into a standalone search string; scope is untouched and any failure falls back to the original question.
- **Provider-agnostic model port:** one OpenAI-compatible adapter covers OpenAI, Groq, DeepSeek and vLLM; errors carry only a status code, never the response body or key.
- **Bounded:** a step budget stops runaway loops, and every path ends in a final status with a safe user-facing message.
- **Known gaps:** answers stream after verification rather than token by token; weak nearest neighbours need a relevance floor; groundedness is not yet checked.
</details>

## Testing and quality

- 1,000+ automated tests: unit, contract (the same suite runs against in-memory and real adapters), security and live-stack integration tests.
- Strict typing (mypy), linting (ruff, including security rules) and 12 import-linter contracts run on every push.
- CI runs the unit suite with warnings as errors and brings up the full Compose stack for integration tests.

## Repository layout

```
apps/        deployable workloads (api, agent-runtime, retrieval-service, ingestion-worker, mcp-servers)
packages/    shared libraries (contracts, agents, retrieval, search, llm-gateway, persistence,
             identity, authorization, ratelimit, messaging, workflows, observability, config)
tests/       unit · contract · integration · security · e2e
docs/        architecture decision records, security invariants, learning log
scripts/     smoke test, retrieval evaluation
evaluations/ versioned evaluation datasets
```

## Documentation

- [Architecture decision records](docs/adr/)
- [Security invariants](docs/security-invariants.md)
- [Learning log](docs/learning-log.md): what was built at each gate, why, and what went wrong
