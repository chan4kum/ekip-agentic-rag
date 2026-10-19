# Learning Log

One entry per approved step: what we built, why, and gotchas.

## Gate 0.1 — ADR-001 modular monorepo (2026-10-19)
- **What:** decision record fixing repo layout (`apps/*`, `packages/*`) and import rules.
- **Why:** written boundaries stop the repo turning into a tangle; ports/adapters make provider swaps cheap.
- **Concepts:** modular monolith, dependency direction, Dependency Inversion.
- **Gotcha:** rules only matter if CI enforces them (import-linter, added in Phase 1).

## Gate 0.2 — ADR-006 SOLID + DI rules (2026-10-19)
- **What:** eight concrete rules: ports as Protocols, constructor injection, one composition root, small interfaces, contract tests per adapter.
- **Why:** providers (Groq, OpenAI, Gemini, OpenSearch, Redis, MCP) will change; business logic must not know which one is in use.
- **Concepts:** Dependency Inversion, composition root, structural typing (`Protocol`), test doubles.
- **Gotcha:** `Depends()` belongs at the HTTP edge only; services must stay framework-free.

## Gate 0.3 — Security invariants (2026-10-19)
- **What:** eleven rules that must always hold (fail-closed, LLM not a boundary, untrusted data, tenant isolation, no token passthrough, bounded agents, audit).
- **Why:** AI systems add attack paths (poisoned documents steering tools); naming the "must never happen" cases up front gives later tests a target.
- **Concepts:** fail-closed, confused deputy, prompt injection, defense in depth.
- **Gotcha:** `tenant_id` must never come from the request body or the model.

---
**Phase 0 complete.** Next: Phase 1, Gate 1.1 (toolchain).

## Gate 1.1 — Toolchain: uv + Python 3.12 + ruff/mypy/pytest (2026-10-19)
- **What:** `pyproject.toml` (deps, ruff, mypy strict, pytest-asyncio), `.python-version`, smoke test, `uv.lock`.
- **Why:** reproducible environments (lockfile), early bug catching (types + lint), async-ready tests.
- **Concepts:** dependency groups, lockfile, strict typing, bandit-style lint rules.
- **Gotchas found:**
  1. Ruff 0.17 also formats Markdown code blocks → `docs/` excluded from ruff.
  2. The mypy pydantic plugin errors if pydantic is not installed → plugin line added in step 1.3, not now.
  3. The sandbox's shared folder can't host mypy's SQLite cache (disk I/O error) → sandbox sets `MYPY_CACHE_DIR` outside the repo; your Mac is unaffected.
  4. Never create `.venv` in the shared folder (Linux binaries); sandbox venv lives outside it.

## Gate 1.2 — Skeleton, .env.example, README v0 (2026-10-19)
- **What:** folder skeleton per ADR-001, `.env.example` (names only), portfolio README with honest status table.
- **Why:** stable layout, documented config without secrets, README from day one.
- **Concepts:** config separation, secret hygiene (I-8), per-provider settings.
- **Gotcha:** the app reads `LLM_BASE_URL`, not the global `OPENAI_BASE_URL`, so OpenAI embeddings can't silently route to Groq.

## Gate 1.3 — Typed settings + first uv workspace member (2026-10-19)
- **What:** `packages/config` (`ekip-config`) with a frozen pydantic-settings `Settings`; 9 unit tests; root project depends on it via `[tool.uv.sources] workspace = true`.
- **Why:** one validated config source; fail fast at startup; secrets masked (`SecretStr`); `LLM_BASE_URL` instead of global `OPENAI_BASE_URL`.
- **Concepts:** 12-factor config, `SecretStr`, model validators, immutability, uv workspace + single lockfile.
- **Gotchas:** with the pydantic mypy plugin on, `Settings(_env_file=None)` needs no `type: ignore`; `load_settings()` belongs at the composition root only (ADR-006 rules 3 and 7).
- **Bug caught by a test:** `PostgresDsn` is a multi-host URL (no `.host`, use `.hosts()`); my first `getattr` loop crashed and mypy could not see it. Replaced with explicit typed access plus a parametrized test for all three URLs.

## Gate 1.4 — Structured logging + request ID (2026-10-19)
- **What:** `packages/observability` (`ekip-observability`): structlog JSON logging, `redact_sensitive` processor, `bind_request_context` / `clear_request_context`; 6 tests.
- **Why:** one JSON line per event, tied to a request id via `contextvars` (safe under concurrency); secrets masked before rendering (I-8, I-11).
- **Concepts:** processor pipeline, contextvars, redaction by key name, logs to stdout (12-factor).
- **Limits:** key-name redaction misses secrets embedded in free text; uvicorn access logs are bridged in 1.5; module-level `get_logger` is a deliberate ADR-006 exception (cross-cutting infra).
- **Decision recorded:** chat LLM = OpenAI (explicit `https://api.openai.com/v1`), embeddings = Google Gemini (needs `GOOGLE_API_KEY` in `.env`), Groq optional fallback. Applied in Phases 3 and 6.

## Gate 1.5 — FastAPI app factory, /health, request-ID middleware (2026-10-19)
- **What:** `apps/api` (`ekip-api`): `create_app(settings)` composition root, lifespan, pure-ASGI `RequestContextMiddleware`, `/health` + `HealthResponse`; 11 tests; verified with a real uvicorn run.
- **Why:** injected settings (ADR-006), request IDs on every log line and response, sanitised incoming IDs (log-injection defence), docs hidden in production.
- **Concepts:** ASGI, app factory (`uvicorn --factory`), lifespan, why not `BaseHTTPMiddleware` (breaks contextvars/streaming), response models.
- **Gotchas:** httpx `ASGITransport` does not run lifespan (test it via `app.router.lifespan_context`); Starlette's `TestClient` now warns about httpx, so it is avoided; unhandled 500s lack the request-ID header until Problem+JSON handling in 2.6.

## Gate 1.6 — First DI port: Retriever + RetrievalService + in-memory adapter (2026-10-19)
- **What:** `packages/contracts` (`Retriever` Protocol, `RetrievedChunk`); `RetrievalService` (validates, framework-free); `InMemoryRetriever` adapter; `GET /api/v1/search`; `create_app(retriever=...)`; 23 new tests incl. a reusable **contract suite**.
- **Why:** service never knows which search engine runs; adapters are chosen only in the composition root; the same service will back agents and an MCP `search_documents` tool.
- **Concepts:** ports and adapters, `Protocol`, constructor injection, `Depends`, `dependency_overrides`, contract tests (Liskov), defense in depth (edge + service validation).
- **Security:** tenant is server-decided (stub `get_tenant_id`, fails closed with 401 outside local/test) and mandatory keyword-only in the port (I-1, I-4). Stub is replaced by the validated JWT context in step 2.3.
- **Deferred:** import-linter rules moved to Gate 1.9 (CI).

## Decision — Guardrails framework: NVIDIA NeMo Guardrails (2026-10-19)
- **Chosen by:** project owner. Role: programmable runtime rails (input/output/dialogue/topic control, RAG hallucination checks) using Colang policies; integrates with LangChain/LangGraph.
- **Placement:** behind a `GuardrailEngine` port in `packages/guardrails` with a `NemoGuardrailsEngine` adapter (ADR-006), so it stays replaceable and unit-testable with a fake.
- **Boundary (I-2, I-6):** NeMo is a defense layer, NOT the security boundary. Authentication, tenant isolation, ACLs, tool allowlists and write-approval stay deterministic code. Rails complement them.
- **Timing:** basic input/output rails can arrive with Phase 6 (RAG answers); full hardening in Phase 9; MCP-specific tool rails in 7A.14. Version/API compatibility with our Python 3.12 + LangGraph stack is verified at that gate before adoption.

## Note — Decision-model judges (e.g. JEV by TypeSafe AI) (2026-10-19)
- **What:** non-generative "System 1" judge: state + typed questions in, typed verdicts + probabilities out. Early access (mid-Sep 2026); access method, speed and price claims are vendor claims, unverified.
- **Adopt the pattern, not the product:** a `Judge` port (typed verdict + confidence) with a cascade: fast judge -> if confidence below threshold -> reasoning-LLM judge or human approval. Candidates behind the port: NeMo Guardrails rails, an LLM judge, JEV once access and calibration are verified.
- **Rules:** threshold (draft 0.90) must be set from measured calibration on our own cases; a judge may only *add* a pause/deny signal, never grant access (I-2, I-6). First use: write-tool argument checks in 7A.10; evals in Phase 10.

## Gate 1.7 — Docker Compose: Postgres 18, Redis 8, OpenSearch 3.9 (2026-10-19)
- **What:** `docker-compose.yml` with healthchecks, localhost-only ports, named volumes, pinned tags; 4 guard tests that parse the YAML.
- **Why:** real infrastructure for later phases; "ready" (healthy) is different from "started"; guard tests stop a later edit exposing a database.
- **Concepts:** Compose services/volumes, healthchecks, `up --wait`, 127.0.0.1 binding, immutable tags, dev-only security-off OpenSearch.
- **Gotchas:** PG18 image volume mount is `/var/lib/postgresql` (not `/data`); the sandbox has no Docker, so container start-up is verified on the owner's Mac.

## Gate 1.7a — Containerise the API (2026-10-19)
- **What:** `apps/api/Dockerfile` (multi-stage, uv, frozen lock, non-root), `.dockerignore` (excludes `.env`), hardened `api` service (read-only rootfs, `cap_drop: ALL`, no-new-privileges, explicit env only, waits for healthy deps); 8 more guard tests.
- **Why:** smaller image and attack surface; secrets can't leak into layers (I-8); image equals what was tested (`--frozen`).
- **Concepts:** multi-stage builds, layer caching, build context in a uv workspace, least privilege, `.dockerignore` as a secrets control.
- **Note:** `--proxy-headers` trust is configured in 1.7b; port 8000 is temporary until Traefik is the only entry.

## Gate 1.7b — Traefik reverse proxy + network segmentation (2026-10-19)
- **What:** Traefik v3.7.15 as the only entry (`127.0.0.1:8080`), file provider, security headers, 1 MiB body limit, 50 rps/IP rate limit; networks `edge`, `app` (internal), `dev`; API has no published port and no internet; trusted-proxy client IP (`FORWARDED_ALLOW_IPS`); `client_ip` in access logs; 10 guard tests.
- **Why:** one controlled front door; spoof-proof client IP; compromised API cannot call out; `/docs` never exposed at the edge.
- **Concepts:** reverse proxy vs gateway, trust boundary for X-Forwarded-For, internal networks, static config vs Docker provider (docker.sock = root-equivalent).
- **Open:** TLS/HSTS later; SSE behaviour through Traefik tested in 2.7; Docker Desktop NAT makes the per-IP rate limit global locally.

## Gate 1.7c — Squid egress (forward) proxy (2026-10-19)
- **What:** `infrastructure/squid/squid.conf` (exact-host HTTPS allowlist, CONNECT-only, private/metadata IP ranges denied, `deny all` last, no TLS interception, no cache); `squid` service on `app` + new `egress` network (non-root, read-only, `cap_drop: ALL`); `api` uses `HTTPS_PROXY` with `NO_PROXY` for internal services; 12 guard tests (83 total).
- **Why:** a prompt-injected or compromised app cannot reach arbitrary hosts or cloud-metadata IPs (I-3); one log of every outbound destination.
- **Concepts:** forward vs reverse proxy, egress allowlisting, first-match ACL order, DNS-rebinding defence via resolved-IP deny, CONNECT tunnelling (no key leakage into logs).
- **Unverified (needs Mac):** `squid -k parse`, non-root start with `cap_drop ALL`, `/dev/stdout` logging as uid 13 (fallback: log file on tmpfs). `6.6-24.04_edge` has no stable channel; digest-pin in Gate 1.9.
- **Added to plan:** Gate 1.7d — Traefik resilience middlewares (retry, circuit breaker, in-flight limit); check jitter support in v3.7 docs first.

## Gate 1.7d — Edge resilience in Traefik (2026-10-19)
- **What:** `retry` (2 attempts, network errors only), `circuitBreaker` (503 when open), `inFlightReq` (50/IP), upstream `forwardingTimeouts`; router order `secure-headers, rate-limit, in-flight, body-limit, circuit-breaker, retry`; 6 more guard tests (89 total).
- **Why:** fail fast instead of piling up on a sick api; absorb one-off connection blips without amplifying load; stop one client occupying all capacity.
- **Concepts:** retry storms, breaker state machine (closed/open/recovering), bulkhead, idempotency; breaker outside retry so it counts final outcomes.
- **Docs-verified (v3.7):** retry has exponential backoff but NO jitter option, so real jitter goes in our own clients (7A.9); 5xx not retried unless `status` is set (deliberately unset); POST/PATCH/LOCK not retried by default.
- **Caveats:** all thresholds are drafts, tuned in Phase 11; a breaker on tiny traffic can trip on one 500; SSE streams hold in-flight slots and retry+SSE is untested (Gate 2.7); per-IP limits are effectively global behind Docker Desktop NAT; per-tenant bulkhead needs the validated token (Phase 2).

## Gate 1.8 — `/ready` readiness probe (2026-10-19)
- **What:** `DependencyProbe` port (contracts); Postgres/Redis/OpenSearch probes; `ReadinessService` (concurrent, 2s timeout per probe, fail closed, empty list rejected); `GET /ready` -> 200/503 with per-check `ok|fail`; `create_app(..., probes=)` for injection; new runtime deps `asyncpg`, `redis`, `httpx`; 14 new unit tests + 2 live integration tests (skipped unless `RUN_INTEGRATION=1`). Totals: 98 passed, 2 skipped.
- **Why:** liveness (`/health`) vs readiness (`/ready`) are different decisions: restart vs stop routing. A hung dependency must not hang the probe; error text can leak hostnames/DSNs (I-8), so only the exception type is logged.
- **Concepts:** ports and adapters again, timeouts via `asyncio.timeout`, `gather` concurrency, fail-closed, `trust_env=False` so internal calls never use the egress proxy.
- **Gotchas:** `postgresql+asyncpg://` must become `postgresql://` for raw asyncpg; asyncpg has no stubs (mypy override); `/ready` deliberately not routed by Traefik (guard test).
- **Unverified (needs Mac):** probes against live services (`RUN_INTEGRATION=1 uv run pytest tests/integration`); rebuilt image with new deps.

## Mac verification of Gates 1.7–1.8 (2026-10-19) — run live on Docker 29.5.3
- **Result:** all 6 services healthy; edge `/health` 200 with security headers + `X-Request-Id`; `/ready` and `/docs` 404 at the edge; 1.1 MB body -> 413; 300 rapid requests -> 281x200, 19x429; `/api/v1/search?q=torque` returns chunks; api `/ready` all ok, with Redis stopped -> 503 `redis: fail`; `RUN_INTEGRATION=1` tests 2 passed.
- **Egress:** allowed host via Squid reaches OpenAI (401 without key = reachable); `example.com` -> 403 Forbidden from proxy; direct connection (proxy unset) -> DNS failure; Squid log shows `TCP_TUNNEL/200` vs `TCP_DENIED/403` with host only. `squid -k parse` OK, non-root + `cap_drop ALL` + stdout logging all work.
- **Resilience:** api stopped -> first request 502, then breaker answers 503 immediately; after restart, 503 for ~35 s then 200 (recovery ramp). Confirms the small-sample caveat: one failure opened the breaker.
- **Bugs found only by running Docker (fixed):**
  1. `ModuleNotFoundError: ekip_api` in the image: `uv sync` installed workspace packages *editable* (paths into `/app/apps/...`), but the runtime stage copies only the venv. Fix: `--no-editable`.
  2. Host port 5432 was taken by another project's container; Postgres host port is now `${POSTGRES_HOST_PORT:-5432}` (we ran on 5433). Other project's container untouched.
- **Known noise:** Squid prints `WARNING: HTTP requires the use of Via` (from `via off`; harmless for CONNECT tunnels) and logs a `NONE_NONE/000` line per 10 s healthcheck probe.
- **Lesson:** structural YAML tests cannot catch runtime problems; keep a live smoke test in CI (Gate 1.9).

## Gate 1.4b — Logging gaps closed (2026-10-19)
- **What:** structlog is now the single pipeline; stdlib logging (uvicorn, httpx, asyncpg, redis) is bridged via `ProcessorFormatter`; every line has `logger`, `service`, `environment`, `request_id`; redaction applies to library logs (incl. stdlib `extra=`); `configure_logging` is idempotent (marks and replaces only its own handler); `apps/api/uvicorn-log.json` + `--log-config` make uvicorn's startup lines JSON; JSON whenever there is no TTY (containers), pretty console only in a developer terminal. 15 new tests; 114 passed + 2 skipped.
- **Why:** before, library lines were plain text with no redaction or request ID; a DSN or URL in a library message could leak (I-8); two lines per event if config ran twice.
- **Concepts:** structlog<->stdlib bridge (`foreign_pre_chain`, `wrap_for_formatter`), handler ownership/idempotency, log injection (JSON escapes newlines), TTY detection.
- **Bugs found by running it in Docker:** (1) `APP_ENV=local` meant colourised console output even in a container; (2) re-propagating `uvicorn.access` revived uvicorn's access log, which prints the full URL *including user query text* — now silenced; our `http_request` line logs the path only; (3) uvicorn's `color_message` duplicate with ANSI codes — dropped.
- **Verified on Mac:** all api log lines are valid JSON (including startup); the search query text appears 0 times in logs.
- **Deferred:** `trace_id`/`span_id` fields arrive with OpenTelemetry in Gate 1.10; log shipping/aggregation (Loki) is Phase 10; audit log is Phase 2.

## Gate 1.9 — CI, architecture rules, pinned images (2026-10-19)
- **What:** `.github/workflows/ci.yml` (job `quality`: frozen sync, ruff, format check, mypy, `lint-imports`, pytest; job `stack`: live Compose up, `scripts/smoke.sh`, integration tests, logs on failure, always `down -v`); 5 import-linter contracts from ADR-001 in `pyproject.toml`; all five Compose images pinned by digest; Dependabot (actions, docker, uv); 11 guard tests. Totals: 125 passed + 2 skipped (integration).
- **Why:** unit tests missed the Dockerfile bug; ADR-001 promised enforcement; mutable tags and action versions are supply-chain risks. Workflow token is read-only, actions are pinned to commit SHAs, `persist-credentials: false`, no secrets used.
- **Concepts:** CI vs CD, least-privilege tokens, SHA/digest pinning, architecture tests (forbidden + layers contracts), smoke test vs unit test, `set -euo pipefail`.
- **Verified:** import-linter: 5 kept / 0 broken on first run; a negative test proves a violating toy project fails; `actionlint` clean; `uv lock --check` ok; smoke script passes against the live stack on the Mac (with pinned digests) and fails when Redis is stopped; integration tests 2 passed.
- **Versions used:** actions/checkout v7.0.1, astral-sh/setup-uv v10.3.0 (SHAs resolved via `git ls-remote`).
- **Not verified:** the workflow has never run on GitHub (no remote yet). Treat the first run as the test of the workflow itself. Squid is an `edge` build: review Dependabot digest bumps, do not auto-merge.
- **Next:** Gate 1.10 minimal OpenTelemetry bootstrap, then Phase 2.

## Gate 1.10 — Minimal OpenTelemetry tracing (2026-10-19) — Phase 1 complete
- **What:** `ekip_observability.tracing` (`build_tracer_provider`, `scrub_url_attributes`, `ScrubbingSpanProcessor`, one-line-JSON console exporter); `trace_id`/`span_id` added to every log line inside a request; FastAPI instrumented in the composition root (provider injected, no globals; app shuts down only a provider it built); `TRACING_EXPORTER=none|console` setting (default `none`; named to avoid the SDK's `OTEL_*` variables); probes `/health` and `/ready` excluded; per-message ASGI `send`/`receive` spans excluded so one request = one span. Pins: otel api/sdk 1.45.x, instrumentation 0.66b1 (beta). Totals: 137 passed + 2 skipped.
- **Why:** correlate logs, spans and later retrieval/LLM/tool spans under one trace; exporters (Langfuse/OTLP collector) are a Phase 10 swap, not a code change.
- **Concepts:** trace vs span, W3C `traceparent` continuation, middleware order (outermost = span exists for inner logs), processors vs hooks, exporter separation.
- **Bug caught by a test:** the instrumentation sets `http.url` (with `?q=<the user's question>`) AFTER the span starts, so a span-processor `on_start` scrubber saw nothing. Fix: also scrub in the instrumentation's `server_request_hook`; the processor still covers attributes set at start (e.g. future outbound HTTP spans, where a provider key may sit in the query string).
- **Verified on Mac:** with `TRACING_EXPORTER=console` one search = one span line; its `trace_id` equals the one in the `http_request` log line; `http.url` is path-only; query text appears 0 times in logs; probes produce no spans; smoke test passes with the default exporter.
- **Not done (by design):** OTLP exporter and collector (Phase 10), metrics (Phase 10), global provider registration (when LangGraph/Langfuse need it, Phase 6), sampling policy (Phase 11).

## Phase 1 summary
- Gates 1.1–1.10 plus 1.4b, 1.7a–d done; 137 tests + 2 live integration tests; CI workflow written (not yet run on GitHub: no remote).
- Bugs found only by running the system: editable-install image crash, uvicorn access log leaking queries, console format in containers, query leakage in span URLs. Lesson: every "no leak" claim needs a test that tries to leak.
