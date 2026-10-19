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
| 2 | Identity and backend: OIDC/JWT, tenancy, authorization, rate limits, streaming | Planned |
| 3 | Single-document ingestion: parse, chunk, embed, index | Planned |
| 4 | Incremental ingestion: events, workflows, idempotency, deletes, ACL changes | Planned |
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
