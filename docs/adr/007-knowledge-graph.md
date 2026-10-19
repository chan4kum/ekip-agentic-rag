# ADR-007: Knowledge graph store (GraphRAG, priority P2)

Status: accepted (2026-10-19). Revisit when GraphRAG enters scope (after Phase 5 evaluation).

## Context
GraphRAG (entity/relationship retrieval, multi-hop) is P2: it is added only if the hybrid retrieval
evaluation (Phase 5) shows questions it cannot answer. Candidate: TigerGraph. Requirement from the
owner: open source; if not, pick a distributed, open-source, Postgres-based option.

## Findings (checked 2026-10-19)
- **TigerGraph is not open source.** The Community Edition is free but proprietary ("object code
  only"): internal use only, no third-party use, no benchmark publication, capped at 300 GB of graph
  plus vector data, no support. Enterprise is a paid licence. It fails the open-source requirement.
- **Apache AGE** (Apache Software Foundation, Apache licence) is a PostgreSQL extension that adds a
  property graph with a Cypher-like language next to plain SQL. Current line 1.8.x supports
  PostgreSQL 18 (1.7 supported 17 and 18). It runs inside the Postgres we already operate.
- **Distribution caveat:** AGE's FAQ states it has been tested with Citus but AGE tables cannot yet be
  distributed with Citus (Citus lacks support for inherited tables). So no Postgres-based graph is
  truly sharded today; AGE scales by partitioning, indexing, tuning and read replicas.
- True horizontally sharded open-source graph engines exist outside Postgres (for example
  NebulaGraph, JanusGraph on Cassandra/Scylla); their licences and fit were NOT verified in this
  review and must be before any adoption.

## Decision
1. Define a `KnowledgeGraph` port in `ekip_contracts` when GraphRAG starts; no graph code before then.
2. First adapter: **Apache AGE on PostgreSQL 18** (open source, same operational footprint, tenant
   isolation by graph-per-tenant plus the same ACL rules as chunks).
3. If evaluation shows a single node cannot hold the graph, swap the adapter for a sharded engine
   behind the same port; selection and licence check happen then.

## Consequences
- No new infrastructure in Phases 3-7A.
- Graph data inherits document ACLs: an edge derived from a document is visible only to those who
  can read that document (to be enforced and tested when built).
