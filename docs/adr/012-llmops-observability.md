# ADR-012: LLMOps and observability

Status: accepted. Phases 10-11.

## Context
The platform needs to answer: how is it behaving now, can this release ship, and what did a given
request do - without leaking tenant data into monitoring systems.

## Decisions

1. **Metrics are Prometheus, low-cardinality, and private to a registry object.** `EkipMetrics`
   owns one `CollectorRegistry` (no globals, tests get fresh ones). Labels are fixed vocabularies:
   route templates, status, outcome, model. Tenant, user, question, document and conversation ids
   are forbidden label names (tested); values outside a bounded vocabulary collapse to `other`.
   Per-request detail belongs in traces and logs.
2. **`/metrics` fails closed.** The pure-ASGI middleware serves it only with a configured bearer
   token (or an explicit `allow_anonymous` for local development) and answers 404 otherwise. It is
   not routed through the application's own auth, so it is independent of identity outages.
3. **Tracing uses OpenTelemetry GenAI conventions behind an `LLMTracer` port.** Attribute names
   are pinned in one class; prompt/completion content is off by default and redacted when on.
   Langfuse receives spans over standard OTLP/HTTP (Basic auth); there is no vendor SDK and the
   exporter attaches after the scrubbing span processor.
4. **Prompts and models are data.** `ops/model-registry.yaml` records each prompt's version and
   sha256. A unit test fails if text changed without a version bump or the registry is stale;
   `scripts/update_registry.py` regenerates it. The expectation: change text -> bump version ->
   run evaluations -> regenerate registry -> commit.
5. **Release gates are thresholds in `ops/eval-gates.yaml`, evaluated by a pure function.** A gate
   whose measurement is missing fails. Security metrics (leak count, injection success) are
   equality-to-zero gates, never traded against quality averages. Thresholds on the small synthetic
   dataset are starting points, not production quality bars.
6. **Canary comparison is deliberately simple:** minimum sample per arm, one-sided non-inferiority
   test on success rates, a p95 ratio for latency, rollback beats hold beats promote.
7. **Alerts implement the masterplan SLOs** (availability burn rates, simple-RAG P95 5 s, first
   token P95 2 s, freshness P95 5 min, DLQ, consumer lag, permission propagation). A test checks
   every alert and dashboard query against the metrics that exist and every runbook link.
8. **Load testing is a local tool with a guard rail:** non-localhost targets need `--i-know`, the
   token comes from an environment variable and is never printed.

## Contracts other components must meet
- Agent evaluation: `scripts/eval_agent.py --json` prints one JSON object of flat numeric keys
  (`agent.task_success`, `agent.groundedness`, `agent.leaks`, `agent.injection_success`).
- Load test: `scripts/load_test.py --json-out` writes `load.*` keys; `llm.cost_usd_per_request`
  comes from the metrics deltas of the same run.
- Emitting code calls `EkipMetrics` methods (`observe_chat`, `observe_retrieval`,
  `observe_agent_run`, `observe_llm`, `ingest_outcome`, `hit/miss/stale/blocked`, ...).

## Consequences / known gaps
- Cost uses a static price table and reports 0 for unknown models rather than guessing.
- Online feedback collection, offline LLM-as-judge scoring and dataset curation from production
  traces are not built; the gates consume whatever the evaluation scripts measure.
- Dashboards and rules are plain files; deploying them (Grafana provisioning, Prometheus rule
  loading) is part of the environment build.
