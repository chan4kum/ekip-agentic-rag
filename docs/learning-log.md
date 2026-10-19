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

## Process decisions (2026-10-19)
- **Approval mode:** after Phase 1 the owner chose "auto approve Recommended": each gate still opens with a what/why/verify note, then is built and tested without waiting; stop only for destructive/irreversible actions, secrets or keys, and choices where the recommended option is not clearly right.
- **Git:** commit, push and pull when each phase completes. No Claude co-author or attribution lines (owner instruction overrides any default). Repo: private `chan4kum/ekip-agentic-rag`. Before each commit: scan staged files for forbidden paths, key patterns and any value from `.env` (file names only are printed).
- **Phase 1 pushed** as commit `14f65a9`; the first real GitHub Actions run succeeded.

## Gate 2.1 — Keycloak with an imported `ekip` realm (2026-10-19)
- **What:** Keycloak 26.8.1 (digest-pinned) in Compose with `infrastructure/keycloak/realm-ekip.json`: clients `ekip-web` (public, PKCE S256 mandatory, exact redirect URIs), `ekip-api` (bearer-only), `ekip-test` (DEV ONLY password grant); mappers put `aud=ekip-api`, `tenant_id` and a flat `roles` claim in the access token; users alice/bob (two tenants) and carol (admin); 5-minute tokens, brute-force protection, no self-registration; `tenant_id` is admin-only in the user profile. 16 realm/compose guard tests. `scripts/smoke.sh` now checks token claims. Host ports are overridable (`POSTGRES_HOST_PORT`, `KEYCLOAK_HOST_PORT`) because the owner's Mac already uses 5432 and 8081.
- **Why:** the Gate 2.2 validator must be tested against tokens a real IdP signed, not ones we forge. Tenant isolation depends on the tenant claim being admin-assigned (I-4).
- **Concepts:** OIDC realm/client/mapper, public vs bearer-only clients, PKCE, audience, issuer (`iss`) and why `KC_HOSTNAME` + `KC_HOSTNAME_BACKCHANNEL_DYNAMIC` keep it stable between browser and container, user-profile permissions as a security control.
- **Verified live (Mac):** tokens carry `iss=http://localhost:<port>/realms/ekip`, `aud=ekip-api`, correct `tenant_id`/`roles`, 300 s lifetime; wrong password -> `invalid_grant`; password grant on `ekip-web` -> `unauthorized_client`; auth request without PKCE -> `invalid_request: Missing parameter: code_challenge_method`; from the app network the discovery document still reports the localhost issuer and the JWKS is reachable at `keycloak:8080`.
- **For Gate 2.2:** the JWKS has two keys, `RS256 / use=sig` and `RSA-OAEP / use=enc`: the validator must pick by `kid`, require `use=sig` and an algorithm allowlist, never trust the token header's `alg`.
- **Caveats:** `start-dev` cannot run read-only (it re-builds at startup); production uses a built image, Postgres and no import. Dev passwords are committed on purpose (fake users, localhost-only).

## Gate 2.2 — JWT access-token validator + JWKS cache (2026-10-19)
- **What:** new `packages/identity` (`ekip_identity`): `CachedJwks` (TTL 10 min, min 30 s between fetch attempts, stale-if-error up to 1 h, single-flight lock, keeps only RSA `use=sig` keys with a `kid`) and `JwtTokenValidator` (size cap, `alg` allowlist `RS256`, `kid` lookup, signature + `exp/iat/iss/aud/sub` required, 10 s leeway, then `tenant_id` shape check and `roles` type check). Contracts in `ekip_contracts.identity`: `TokenValidator` port, frozen `AccessTokenClaims`, `InvalidTokenError(reason)` with 12 fixed machine-readable reasons. New settings `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URL`; production refuses non-https or localhost values. Compose passes them to the API and adds `keycloak` to `NO_PROXY` (JWKS fetch is internal and must bypass Squid). Import-linter: identity never imports apps; contracts never import identity.
- **Why:** a valid signature from the right issuer is the only thing that lets us trust `tenant_id` (I-4). Every shortcut here is a known real-world attack: `alg=none`, RS256->HS256 key confusion (public key used as HMAC secret), token-supplied `jku`/`x5u` key URLs, unlimited unknown-`kid` refetches (request amplification against the IdP), and tokens in logs.
- **Concepts:** algorithm allowlist (we pick the algorithm, the token only claims one), JWKS rotation via `kid`, stale-if-error vs fail-closed, single-flight, clock leeway, generic 401 for clients vs specific reason for logs (the reason is logged, the token never is).
- **Verified:** 50 new unit tests (valid, expired, nbf, wrong aud/iss, tampered payload, wrong key with a known kid, `alg=none`, HS256 confusion, missing/unknown/overlong kid, enc-only key, `jku`/`x5u` ignored, missing/invalid tenant, bad roles, oversized, garbage, token never in logs; cache TTL, rotation, unknown-kid flood limited to <=2 fetches, IdP down at start, stale served then expires, empty key set, 20 concurrent misses = 1 fetch) + 7 live tests with real Keycloak tokens (alice `tenant-dev`, bob `tenant-other`, carol admin, tampered token rejected, wrong audience, wrong issuer). Totals: 218 passed, 0 skipped on the Mac with the live stack; ruff, mypy, lint-imports clean.
- **Bug found by running, not reading:** after a failed first fetch, the second call inside the 30 s window raised `unknown_key` instead of `key_unavailable`. Matters because 401 "your token is bad" and 503 "we cannot verify right now" are different operational signals. Fixed: with no key ever loaded the cache reports unavailable.
- **Mac gotcha:** macOS put the `hidden` flag on the new editable-install `.pth` file and Python silently skips hidden `.pth` files, so `import ekip_identity` failed only in the Mac venv. Fix: `chflags nohidden .venv/lib/python3.12/site-packages/ekip_identity.pth`. Not a repo bug (CI and the Docker image use `--no-editable`).
- **Not done (by design):** nothing calls the validator yet (Gate 2.3 wires it into a request-scoped `SecurityContext`); no token revocation or introspection (access tokens live 5 min); issuer/JWKS multi-tenant IdPs; EC/PS256 keys (add to the allowlist deliberately if an IdP needs them).
- **Next:** Gate 2.3 `SecurityContext`, replacing the temporary `get_tenant_id` stub.

## Gate 2.3 — SecurityContext as a request-scoped dependency (2026-10-19)
- **What:** `SecurityContext` (subject, tenant_id, roles, authorized_party; frozen) in `ekip_contracts.identity`; `get_security_context` dependency in the API: bearer token -> `TokenValidator` -> context. `create_app(token_validator=...)` is the injection point; by default the composition root builds `JwtTokenValidator` + `CachedJwks` from `OIDC_*` settings (no network call at startup; keys are fetched on the first request). The temporary `get_tenant_id` stub and its "local/test = tenant-dev" bypass are deleted: there is now no unauthenticated path to tenant data in any environment. `/health` and `/ready` stay public. `bind_principal` adds `tenant_id` and `subject` to every later log line of the request.
- **Why:** I-4 (tenant from the validated token only) and I-1 (fail closed). Previously the tenant was a server-side constant; now it is proven per caller.
- **Behaviour:** no/other scheme/empty bearer -> 401 `WWW-Authenticate: Bearer`; any invalid token -> 401 `Bearer error="invalid_token"` with an identical body for every failure reason (no oracle); IdP keys unobtainable -> 503 (operators can tell "bad token" from "cannot verify"); unexpected validator bug -> 500, never an implicit allow; tokens in the query string are ignored; 401/503 carry `Cache-Control: no-store`.
- **Verified:** 21 new tests (all failure kinds return the same body; spoofing tenant via query, `X-Tenant-ID` and a second header does not change the result; alice and bob each see only their own tenant; token never in logs; rejection reason is logged). Smoke test, run through Traefik with real Keycloak tokens, now checks 401 without/with a bad token, alice sees only `dev-*`, bob (even asking for tenant-dev) only `oth-*`. The container fetched keys from `keycloak:8080` (bypassing Squid via `NO_PROXY`) and matched the `localhost:<port>` issuer. Container logs: 0 occurrences of `eyJ`, `tenant_id` present on authenticated requests, rejection reason `malformed` logged for the garbage token. Totals: 239 passed, 0 skipped on the Mac.
- **Concepts:** authentication vs authorization (this gate is only authentication), RFC 6750 challenges, uniform error bodies, composition-root defaults vs test injection, contextvars across ASGI (pure ASGI middleware has no task boundary, so a value bound in a dependency is visible to the access log).
- **Mac gotcha (again):** any `uv sync` that rewrites editable `.pth` files can leave the macOS `hidden` flag on them and Python then silently skips them. Cure: `chflags -R nohidden .venv`. Worth a README note.
- **Next:** Gate 2.4 `Authorizer` port + RBAC (authentication proves who; authorization decides what).

## Gate 2.4 — `Authorizer` port + fail-closed RBAC (2026-10-19)
- **What:** `ekip_contracts.authorization` (`Action` enum, `Resource` with the OWNER tenant, `Decision` with a fixed reason set, `Authorizer` port). New package `ekip_authorization` with `RbacAuthorizer` and a small reviewed `DEFAULT_POLICY` (`ekip-user`: search/read; `ekip-admin`: plus manage). API: `require_permission(Action.X)` is a route dependency that returns the `SecurityContext`; the search route now uses it. `create_app(authorizer=...)` injects a different policy (OpenFGA/OPA later) without touching routes.
- **Why:** authentication says who; this says what. A signed token with no (or an unknown) role must not get in; a tenant admin must not cross tenants.
- **Rules enforced in code and tests:** tenant check runs first and for every role, admins included; role names are exact (no case folding, no trimming); unknown action -> deny; any exception inside policy code -> deny `policy_error`; the policy dict is copied so later mutation cannot widen it; denial -> 403 with an identical generic body, reason/action/tenant/subject only in the log; authorizer unreachable -> 503, never an implicit allow; 401 (authentication) takes precedence over 403.
- **Verified:** 31 new tests (role x action matrix, cross-tenant for user and admin, unknown action string, runtime policy fault, policy copy, every Action granted to some role, injected spy sees the token-derived tenant, deny-all, outage). Mac: image rebuilt with the new package, smoke test passes, 269 passed, 0 skipped; import-linter 5 kept.
- **Test lesson:** my first "policy fault" test passed a broken mapping to the constructor, but the constructor copies it into a plain dict, so the fault never ran and the test failed. The test was wrong, not the code; the fault is now injected after construction.
- **Not done (by design):** per-document ACLs (Phase 4/5: source-system permissions synchronised into the index), relationship-based checks (OpenFGA), ABAC attributes, policy loaded from a file/registry, an audit sink for denials (log only for now).
- **Next:** Gate 2.5 Redis rate limiting (per user and per tenant).

## Gate 2.5 — Redis token-bucket rate limiting (2026-10-19)
- **What:** `RateLimiter` port + `RateLimits` bundle (user, tenant, fail_open) in `ekip_contracts.ratelimit`; new package `ekip_ratelimit` with `RedisRateLimiter` (one atomic Lua script, Redis `TIME` as the clock, idle buckets expire after 2x refill time) and `InMemoryRateLimiter` (same semantics, injectable clock). API: `limit_requests("search")` dependency, listed after `require_permission` so 401/403 requests never spend budget. Per-user bucket (default burst 20, 60/min) and per-tenant bucket (burst 200, 600/min); bucket keys are `tenant:scope:u:<sha256(subject)[:16]>` so nothing from a token shapes a Redis key. 429 + `Retry-After` + `RateLimit-Limit/Remaining`; Redis down -> 503 (`RATE_LIMIT_FAIL_OPEN=true` flips it). Settings are `RATE_LIMIT_*`; the composition root builds Redis clients lazily and closes them at shutdown.
- **Why:** Traefik limits per IP, which is wrong for users behind one corporate NAT and blind to tenants. Once requests cost LLM tokens, a per-user and per-tenant budget is also a cost control and a noisy-neighbour (bulkhead) control.
- **Concepts:** token bucket vs fixed/sliding window (bursts allowed, steady rate bounded, no boundary double-burst), atomicity (Lua) vs check-then-set races, server-side time vs client clocks, fail-open vs fail-closed for a non-security control with a cost impact, response headers for well-behaved clients, parity ("contract") tests: one spec run against both implementations.
- **Verified:** 8 behaviour tests run against both limiters (burst, retry-after, key isolation, refill, capacity cap, cost > remaining not consuming, 50 concurrent callers never over-grant) + exact fake-clock test + Redis-down -> `RateLimitUnavailableError` + TTL expiry live + 10 API tests (429/headers, per-user and per-tenant isolation, unauthenticated/forbidden requests spend nothing, key hygiene, fail-closed/open). Mac with live Redis: 296 passed, 0 skipped; smoke test hammers as carol through Traefik and sees 429 + `Retry-After`, never a 5xx.
- **Not done (by design):** token/cost-based limits for LLM calls (Phase 6/10: tokens per minute and a daily budget per tenant), per-route tuning, distributed-limit metrics (Phase 10), Redis Cluster key tags (single node for now).
- **Next:** Gate 2.6 RFC 9457 Problem+JSON errors, request deadlines/cancellation, idempotency keys.

## Gate 2.6 — Problem+JSON errors, request deadlines, idempotency keys (2026-10-19)
- **2.6a Errors:** `ekip_api/errors.py` turns every error into RFC 9457 `application/problem+json` (`type` = `urn:ekip:problem:<slug>`, `title`, `status`, `detail`, `request_id`). Validation errors keep only `loc/msg/type` (FastAPI's default echoes the caller's `input`; we strip it, tested with a 1,100-char "secret"). Unhandled exceptions give a generic 500 that still carries `X-Request-ID`: the request id is stored on the ASGI scope by `RequestContextMiddleware` because Starlette's outermost `ServerErrorMiddleware` runs after our middleware has cleared its context vars. `ProblemError` gives a route its own problem type.
- **2.6b Deadlines:** `RequestDeadlineMiddleware` (`asyncio.timeout`, default 30 s, setting `REQUEST_TIMEOUT_S`, capped at 55 s) cancels a slow handler, runs its `finally` blocks and answers a 504 problem; it is inner to the request-context middleware so the 504 is logged and carries the id. A test asserts the app deadline stays below Traefik's 60 s `responseHeaderTimeout` (otherwise callers would see opaque proxy errors). A `TimeoutError` raised BY a handler is not mistaken for the deadline (`deadline.expired()`).
- **Found by testing:** once response headers are sent a 504 is impossible. We stop without sending the final body message, so the server drops the connection and the client can tell the stream was truncated (sending a clean terminator would hide truncation). httpx's in-process transport refuses to model that, so the test drives the ASGI interface directly.
- **2.6c Idempotency:** `IdempotencyStore` port + `IdempotencyService` (reserve -> run -> store; replay returns the exact stored status/body with `Idempotent-Replayed: true`; a duplicate in flight -> 409 + `Retry-After`; same key with a different request -> 422 `idempotency-key-reused`; failures, cancellation, 5xx and bodies > 256 KiB release the key instead of freezing it; 4xx are cached because they are deterministic). Keys are scoped by tenant + subject + route and hashed. Stores: in-memory (injectable clock) and Redis (`SET NX PX`, 24 h results, 60 s lock which must stay above the 30 s deadline). Fingerprint = length-prefixed method + path + body.
- **Not mounted yet:** no POST route changes state yet, so it is exercised through a test-only route and gets wired to the first real POST (Phase 3 ingestion trigger, Gate 7A.10 ticket creation). Settings: none yet.
- **Verified:** 53 new tests (store semantics run against memory and live Redis, 25 concurrent `begin`s have exactly one winner, lock expiry, service rules, HTTP behaviour including cross-user/tenant key privacy and 401 never touching the store). Mac with live Redis: 339 passed, 0 skipped; smoke test now also checks problem+json and the echoed request id through Traefik.
- **Next:** Gate 2.7 SSE `/api/v1/chat` skeleton. Open question to test live: does Traefik's buffering middleware hold SSE back? (it buffers responses).

## Gate 2.7 — SSE `/api/v1/chat` skeleton (2026-10-19)
- **What:** `POST /api/v1/chat` streams Server-Sent Events: `meta` (request id, conversation id) -> `token`* -> `citations` -> `done`; mid-stream problems arrive in-band as `error` (generic, no internals). Behind a `ChatStreamer` port (`ekip_contracts.chat`); the Phase-2 `StubChatStreamer` only restates retrieved passages (no model) and is replaced by the LangGraph agent in Phase 6. `ChatRequest` forbids unknown fields (a caller-sent `tenant_id` is a 422, never silently ignored). Auth, authorization and rate limiting run before the stream starts, so those failures are ordinary 401/403/429 problem+json. Own deadline (`CHAT_TIMEOUT_S`, 120 s) and 15 s heartbeats; the generic 30 s request deadline exempts the route.
- **Streaming internals:** `with_heartbeats` runs the source in its own task behind a bounded queue (backpressure), emits `: keep-alive` comments when quiet, enforces the overall deadline, and on consumer cancellation (client disconnect) cancels and awaits the source so its `finally` runs. A mutation check (remove the cancellation) made exactly the three cleanup tests fail, so they do detect the bug. SSE framing: event names are a closed set; user text only travels inside single-line JSON, so a message containing `\n\nevent: done` cannot forge events (tested).
- **Found live, not by unit tests:** Traefik's `buffering` middleware (our request body limit) also buffers whole RESPONSES: through the edge the first token and `done` arrived together after the whole answer. Fix: a dedicated `chat` router (priority 100) without `buffering`/`retry` (own in-flight budget `in-flight-chat`), plus an in-app `MaxBodySizeMiddleware` (`MAX_REQUEST_BODY_BYTES`, default equal to the edge's 1 MiB; rejects on Content-Length before reading, and cuts chunked uploads at the limit; 413/400 problem+json). Smoke test now fails if the stream is buffered (first token vs done must differ by >= 1 s) and if an oversized chat body is accepted. Guard tests keep `buffering`/`retry` off the chat router and `writeTimeout` off the entrypoint.
- **Verified:** 29 new tests + 3 proxy guards; Mac: smoke "first token 0.00 s, done 3.34 s" through Traefik, 413 on a 1.1 MB chat body, 380 passed, 0 skipped.
- **Caveats:** per-IP limits (rate-limit, in-flight) behave globally behind Docker Desktop/corporate NAT; the stub's typing delay (`CHAT_STUB_TOKEN_DELAY_S`, 0.3 in dev compose) exists only to make streaming observable; reconnect/resume (`Last-Event-ID`) is not implemented (Phase 6 with checkpoints); HTTP/2 and TLS termination are not configured.
- **Next:** Gate 2.8 security regression suite, then Phase 2 close-out (README, secret scan, commit, push, pull, CI check).

## Gate 2.8 — Security regression suite (2026-10-19)
- **What:** `tests/security/test_authentication_attacks.py` (65 tests) and `test_live_keycloak.py` (6 tests against real Keycloak tokens). A **route inventory** built from the OpenAPI schema fails any new route until it is added to the PUBLIC or PROTECTED list (and so to every attack below), and fails if a route is hidden from the schema. Each protected route (`GET /api/v1/search`, `POST /api/v1/chat`) is attacked with 20 credentials: expired, not-yet-valid, wrong audience (also a list without ours), wrong issuer, `alg=none` (also claiming admin), HS256 key confusion (signed with the public key as secret), foreign signing key (also claiming admin), unknown `kid`, tampered tenant/roles without re-signing, missing tenant/subject, path characters in tenant, `jku` header, garbage, empty segments, oversized. All must return the **same** 401 problem body, so a caller learns nothing about which check fired. Also: no credentials -> 401 before body validation (never 422), non-Bearer schemes / cookies / `access_token` query param ignored, 10 s leeway boundary, roles that look like admin but are not (`EKIP-ADMIN`, trailing space) -> 403, spoofed identity headers (`X-Tenant-ID`, `X-Roles`, ...) change nothing, cross-tenant matrix for both routes even when the caller asks for the other tenant (header, query, body), no token segment ever appears in captured logs, production wiring with an unreachable IdP never returns 200 and hides `/docs`.
- **Found while writing:** this FastAPI version returns lazily wrapped included routers from `app.routes` (no `APIRoute` objects), so a naive route walk saw nothing and the inventory passed vacuously. The first run failed on the second assertion ("classified routes must exist"), which is why that assertion is there. Lesson: a security inventory test needs a "the inventory is not empty" check.
- **Mutation checks (do the tests detect regressions?):** signature verification off -> 21 failures; issuer unchecked -> 2; audience + tenant unchecked -> 18; leeway 3600 s -> 3. Removing only our own `alg` allowlist check changed nothing, because PyJWT enforces the same allowlist in `decode` and refuses HMAC with an RSA key: two independent layers, so the suite cannot tell them apart (accepted, defence in depth).
- **Verified:** sandbox 422 passed / 29 live-only skipped; Mac with live Keycloak, Redis, Postgres, OpenSearch: 451 passed, 0 skipped; smoke passes (SSE first token 0.00 s, done 3.33 s through Traefik); ruff, mypy (83 files) and import-linter (5 contracts) clean.

## Phase 2 summary
Request path now: Traefik (per-IP limits, body cap, security headers) -> request context (id on the ASGI scope) -> body limit -> deadline -> JWT validation (RS256 allowlist, cached JWKS, tenant claim required) -> RBAC with tenant check first -> per-user and per-tenant token bucket (Lua on Redis, fail closed) -> handler; errors are problem+json; SSE streams with heartbeats and cleanup on disconnect. Tenant identity never comes from the request, only from the verified token (I-4).
- **Known gaps carried forward:** TLS/HSTS at the edge, per-IP limits behave globally behind NAT, SSE reconnect (`Last-Event-ID`), idempotency not yet mounted on a real POST route (first user: Phase 3 ingestion trigger), token/cost budgets for LLM calls (Phase 6/10), OpenFGA for document-level ACLs (Phase 4/5), Keycloak runs in `start-dev` locally.
- **Next:** Phase 3, single-document ingestion (contracts, Postgres repositories with Alembic, object store, parser, chunker, embedder, OpenSearch indexer).

## Gate 3.1 — Document, chunk and embedding contracts (2026-10-19)
- **What:** `ekip_contracts.documents`: `Document` (one immutable VERSION of a source document), `DocumentAcl`, `Chunk`, `ChunkEmbedding`, `DocumentStatus`, and the derivation helpers `document_id_for`, `chunk_id_for`, `sha256_hex`. All frozen dataclasses that validate in the constructor, so an invalid object cannot exist downstream.
- **Why these shapes:** (1) `document_id` = `doc_` + sha256 of the length-prefixed (tenant, source, source id): stable across re-syncs, tenant-namespaced (two tenants never collide, ids cannot be forged: a mismatching id is rejected), unambiguous at field boundaries. (2) `version` is OUR monotonic sequence per document (what Phase 4 compares to stop version 7 overwriting 8); `source_version` is whatever the source calls it. (3) `chunk_id` = `<document_id>:v<version>:c<ordinal>`, so versions coexist during safe publication; `content_hash` is the same for identical text across versions, which is what lets Phase 4 reuse embeddings. (4) Vectors live in `ChunkEmbedding`, tagged with `model_version` (e.g. `gemini-embedding-001@768`): vectors from different models are not comparable, so the tag lets us detect mixed generations. (5) ACL is fail-closed: empty means nobody, group names match exactly. (6) Only `INDEXED` versions are `is_searchable`.
- **Verified:** 51 unit tests (derivation determinism, tenant and boundary collision cases, forged ids, tampered chunk text/ordinal, naive datetimes, NUL bytes, NaN/inf vectors, ACL matching). ruff, mypy, import-linter clean.
- **Not done (by design):** persistence (3.2), raw-bytes storage (3.3), ACL ingestion from sources (Phase 4), per-tenant embedding-model registry (Phase 5).
- **Next:** Gate 3.2 Postgres repositories with Alembic.

## Gate 3.2 — Document repository: ports, in-memory twin, Postgres adapter, Alembic (2026-10-19)
- **What:** `ekip_contracts.document_store` (the `DocumentRepository` port + `RepositoryError` family) and legal status transitions in `ekip_contracts.documents` (`can_transition`, `statuses_that_may_move_to`); new package `ekip_persistence` with `InMemoryDocumentRepository`, `PostgresDocumentRepository` (asyncpg) and Alembic migration `0001` (`documents`, `chunks`). `Document` gained `failure_reason`.
- **Design points:**
  - Every method takes `tenant_id` and every SQL statement filters on it; a row from another tenant looks exactly like a missing row (tested for get, latest_version, transition, replace_chunks, get_chunks).
  - `register` is `INSERT .. ON CONFLICT DO NOTHING RETURNING`: the database decides who wins, a retry with identical input returns the stored row WITHOUT resetting its progress, different content for the same (document, version) is a `VersionConflictError`.
  - `transition` is ONE statement, `UPDATE .. WHERE status = ANY(legal sources) RETURNING`: no read-then-write window. 10 workers racing for the same step: exactly 1 wins, 9 get `InvalidTransitionError` (tested on both adapters). A failed version may resume at any later stage (retry only indexing after good embeddings); INDEXED may only go to DELETED; DELETED is terminal.
  - `replace_chunks` runs in a transaction behind `SELECT .. FOR UPDATE` on the version row, so a crash leaves no partial set and concurrent writers serialize (final state is one writer's complete set, never a mixture). Chunk sets of INDEXED/DELETED versions are frozen (`ImmutableVersionError`). A chunk from another tenant/version in the set rejects the whole call.
  - The migration repeats the contract rules as CHECK constraints (tenant id format, `doc_` ids, sha256 hex, status values), a composite FK from chunks to documents with ON DELETE CASCADE, and indexes for source lookup, in-flight work and chunk-hash lookup (embedding reuse in Phase 4). A test proves the database itself refuses a malformed tenant id and an unknown status.
  - Alembic reads `DATABASE_URL` (no credentials in files; refuses to fall back to the dev default when `APP_ENV=production`); migrations are frozen SQL, independent of application code.
- **Verified:** one spec of 24 behaviours run against both adapters; migration up -> down -> up on a throwaway database; Mac with live Postgres: 559 passed, 0 skipped; sandbox 500 passed / 59 live-only skipped; ruff, mypy, import-linter (6 contracts: persistence knows only contracts) clean. CI's stack job now also runs `tests/contract` against the live database.
- **Not done (by design):** Row-Level Security (needs a non-superuser application role, which the dev image does not have; Phase 9/11), connection-pool wiring into the API lifespan (first use: Gate 3.8 ingestion use case), `image_assets` table (Gate 3.3/3.4 with the object store), version-number allocation (callers read `latest_version` + 1 and a clash surfaces as `VersionConflictError`; Phase 4 adds the retry loop).
- **Mac gotcha refined:** `chflags -R nohidden .venv` sometimes misses a `.pth` (we saw `ekip_identity.pth` stay hidden); use `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` and `uv run --no-sync` afterwards.
- **Next:** Gate 3.3 `ObjectStore` port (raw documents and image assets, content-addressed keys, tenant-prefixed) with local-filesystem and S3-compatible adapters.

## Gate 3.3 — Object store, image assets, image chunks (2026-10-19)
- **What:** `ObjectStore` port (`put/get/head/delete`) with `InMemoryObjectStore` and `LocalFileObjectStore`; `ObjectKey` = (tenant, kind raw|assets, sha256) with path `tenants/<tenant>/<kind>/<sha[:2]>/<sha>`; `ImageAsset` contract (bytes live in the store, metadata in Postgres `image_assets`: page, bbox as page fractions, nearby chunk ordinal, mime, size, dimensions, caption, alt text, OCR text, model description + `description_model_version`); `AssetRepository` port (+ both adapters); `Chunk.kind` (`text|image`) + `asset_sha256`; migration `0002` (table, CHECKs, `kind`/`asset_sha256` coupling constraint).
- **Design points:** (1) The store computes the key FROM the bytes, so a caller cannot mislabel an object, and a key can only be built from a validated tenant id, a closed kind and a sha256: no path-traversal surface at all (hostile tenants/hashes are rejected at construction; the file adapter additionally checks the resolved path stays under its root). (2) Dedupe is per tenant only: identical bytes in two tenants are two objects, so there is no cross-tenant existence oracle and deleting one tenant's copy cannot affect another. (3) Atomic writes (temp file + fsync + rename), integrity re-checked on read (`ObjectIntegrityError`), size limit, first content-type wins on dedupe. (4) An image becomes searchable through an `image` chunk whose text is `ImageAsset.search_text` (caption + alt + OCR + description); the chunk points to the asset by hash, retrieval stays text-only, citations can link to the image. (5) Assets freeze with the version (INDEXED/DELETED), exactly like chunks.
- **Verified:** 12 object-store behaviours on both adapters (+ corruption, layout, no temp files, 20 concurrent identical puts), 25 contract unit tests (hostile keys, asset validation, chunk kind coupling), 6 asset-repository behaviours on both repository adapters, migration up/down/up now includes `image_assets`. Mac with live Postgres: 637 passed, 0 skipped.
- **Not done (by design):** S3/MinIO adapter and presigned URLs (cloud phase; same port and tests), serving images through the API (needs the ACL check from Phase 5), malware scanning of uploads (Phase 9).
- **Next:** Gate 3.4 parser port: text/Markdown/HTML/PDF/DOCX into structured blocks, extracting images, with hostile-input limits.

## Decision — knowledge graph store (2026-10-19, ADR-007)
- Asked: is TigerGraph open source? **No**: Community Edition is free but proprietary (internal use, 300 GB cap, no benchmarking, no support). Chosen: **Apache AGE on PostgreSQL 18** behind a future `KnowledgeGraph` port, introduced only when GraphRAG (P2) is justified by evaluation. Caveat recorded: AGE tables cannot yet be sharded with Citus, so it scales by partitioning/replicas; a sharded engine (e.g. NebulaGraph, JanusGraph) is the fallback behind the same port, licence to be verified then.

## Gate 3.4 — Parser port and parsers (2026-10-19)
- **What:** `ekip_contracts.parsing` (`DocumentParser` port, `ParsedDocument/ParsedBlock/ParsedImage`, `BlockKind`: heading, paragraph, list_item, table, code, image) and new library `ekip_ingestion` (no network, no database) with parsers for text, Markdown, HTML, PDF (pypdf) and DOCX (python-docx), a MIME registry, `ParseLimits`, and safe image inspection.
- **Security design (documents are untrusted input):** hard limits on bytes, pages, blocks, images, table cells and ZIP expansion (DOCX is checked BEFORE opening: zip-bomb guard); magic-byte checks so a DOCX claiming to be a PDF is refused; encrypted PDFs refused; every library failure becomes a generic `ParseError`. HTML: scripts/styles/frames/svg and comments dropped, and text a human cannot see (hidden attribute, aria-hidden, display:none, visibility:hidden, zero font-size/opacity) is dropped because hidden text is the classic carrier for indirect prompt injection (I-3); `<img src="http://...">` is NEVER fetched (no SSRF), only inline data: URIs are read. Images: header-only inspection (a decompression bomb is rejected by its declared pixels before any decoding), SVG refused (XML with scripts/external refs), allow-list PNG/JPEG/GIF/WEBP.
- **Cost design (fewer model calls):** PDF pages without a text layer are reported in `pages_needing_ocr`, so OCR runs only where needed and never on a text PDF; a cheap filter drops icons/bullets/spacers (min side 48 px, min 1 KB) before any vision call; an image repeated across pages (logo) is kept once; HTML/DOCX duplicates are dropped by sha256.
- **Verified:** 45 parser tests (structure, hidden-text variants, external image never fetched, scanned and blank PDFs, repeated images, encrypted/corrupt/wrong-magic files, zip bomb, tiny images, registry routing). Hand-built minimal PDF, python-docx and Pillow generate all fixtures (no binaries in the repo). Mac: 682 passed, 0 skipped.
- **Not done (by design):** layout-aware PDF structure (headings from font sizes/outlines), OCR itself (Gate 3.6 provider), XLSX/PPTX/email parsers (later), running parsers in a separate sandboxed process with a hard timeout (threads cannot be killed; Phase 9/11).
- **Next:** Gate 3.5 normalizer and structure-aware chunker (stable ids, hashes, section paths, image chunks, table/code handling).

## Gate 3.5 — Normalizer and structure-aware chunker (2026-10-19)
- **What:** `ekip_ingestion.chunking`: `strip_boilerplate` (running headers/footers repeated on >= 3 pages and >= half of the text pages, bare page numbers), `chunk_document` (-> `ChunkingResult(chunks, assets)`), `embedding_text` (title + section path + content), `ChunkingConfig` (target 2,000 chars ~500 tokens, ceiling 3,200 ~800 tokens, 200-char overlap only inside one split paragraph), `ImageEnrichment` (OCR text / vision description + model version, looked up by image sha256).
- **Rules:** a chunk never crosses a heading (section boundaries are chunk boundaries); paragraphs and list items are packed to the target; tables and code are their own chunks; an oversized table splits by rows WITH the header repeated, code splits at lines (an endless line is cut), prose splits at sentences with overlap, unpunctuated text at whitespace, an unbroken token is hard-cut. An image becomes a `PendingAsset` (asset + bytes for the object store) in document order, plus an `image` chunk only when it has text (caption/alt/OCR/description); otherwise it is stored and becomes searchable once an enrichment exists.
- **Why these choices:** (1) Determinism: pure function, no model calls; the same input yields identical ids/hashes. (2) Edit stability: because sections are hard boundaries, editing one section leaves every other section's chunk hashes unchanged (tested), which is what lets Phase 4 skip re-embedding; the honest limit is that an edit INSIDE a section can shift that section's own boundaries. (3) Context without an LLM: `embedding_text` prepends title and section path to the text that is embedded, while the stored `content` stays clean for display and citation (a "contextual retrieval" LLM pass is a later experiment, not a default). (4) Enrichment is an input, so the pipeline order is parse -> (batched, cached vision/OCR only for images that survived the filter) -> chunk, and the chunker itself never calls a model.
- **Verified:** 35 tests (section paths, packing with no loss/duplication, oversize paths, determinism, edit stability, 40 random documents always give valid bounded chunks, images with/without text and with enrichment incl. provenance rule, boilerplate thresholds, config validation, DOCX end to end).
- **Limits (recorded):** size is measured in characters (about 4 chars per token), not real tokens, to avoid a tokenizer dependency; PDF headings are not detected (no layout analysis), so PDF chunks carry no section path beyond what bookmarks could add later.
- **Next:** Gate 3.6 `Embedder` port, Gemini adapter (batching, retries, validation), embedding cache and a deterministic fake.

## Gate 3.6 — Embedder port + Gemini provider

**What:** `Embedder` Protocol in `ekip_contracts.embedding` (`model_version`, `dimensions`, `embed(texts, task)`); adapters in `ekip_llm_gateway`: `GeminiEmbedder` (httpx, `batchEmbedContents`), `CachingEmbedder` (+ `EmbeddingCache`, in-memory LRU), `HashingEmbedder` (offline, deterministic). Settings: `embedding_provider/model/dimensions`.

**Why:**
- `model_version` (e.g. `gemini-embedding-001@768`) is stored with every vector; changing model or dimension means a re-embed, never a silent mix.
- `task` (document vs query) matters: Gemini embeds them differently for retrieval.
- Fewer calls = lower latency/cost: in-call dedupe, batches of ≤100 run concurrently, cache keyed by `(model_version, task, sha256(text))`. Unchanged chunks are never re-embedded.
- Truncated (768-dim) vectors are not unit length, so we L2-normalise; cosine == dot product.

**Security:** key only in the `x-goog-api-key` header; errors carry the HTTP status, never the response body or input text; responses validated (count, dimension, finite, non-zero).

**Resilience:** retry 429/5xx/network with jittered exponential backoff, honouring Retry-After (capped 30 s); 4xx is not retried; exhaustion → `EmbeddingUnavailableError`.

**Verified:** 19 unit tests (MockTransport); live test (`RUN_LIVE_LLM=1`) on the Mac returned 768-dim unit vectors with the related document ranked first. Import-linter: 8 contracts kept.

**Gotcha:** `ekip_config` exports `Settings` from `ekip_config.settings`, not the package root.

## Gate 3.7 — OpenSearch mapping + chunk indexer

**What:** `ChunkIndex` port (`ekip_contracts.search_index`: `ensure_ready`, `replace_document`, `delete_document`, `update_acl`, `count_chunks`, `IndexEntry`) and `ekip-search` package: `mapping.py` (pure), `queries.py` (`tenant_and_acl_filter`), `opensearch.py` (httpx adapter).

**Design decisions**
- One shared index per embedding `model_version` (`ekip-chunks-<slug>`); `tenant_id` on every doc, mandatory in every query, and used as the routing key (queries touch fewer shards). A model change = a new index, built beside the old one, so vectors from different models never mix. Existing index with a different mapping version / model / dimension → fail loudly (`ensure_ready`).
- `dynamic: strict`: unknown fields are errors. `source_uri`/`content_hash` are stored, not indexed.
- Vectors: HNSW, Lucene engine, cosine. Lucene filters *during* graph traversal, so a restrictive ACL filter still returns k hits (verified live: the 5 nearest neighbours were forbidden docs and the allowed doc was still found).
- ACL fail-closed by construction: a doc listing nobody has empty `allowed_*` fields and matches neither filter clause; there is no "public" flag to forget.
- `replace_document`: reject stale (older version never overwrites newer, via max-version aggregation); bulk in batches of 200 with `_id = chunk_id` (idempotent); then delete older-version and leftover same-version chunks; return the live count so the caller can validate the publication. Failed bulk items fail the whole publication.
- `update_acl` rewrites ACL fields in place (painless script, no re-embedding): the Phase 4 permission-change path.
- Errors carry the HTTP status only (response bodies can echo document text); 429/502/503/504 and network errors retried with backoff.

**Gaps (deliberate, later phases):** no "invisible until validated" flag between bulk and prune (a brief overlap of old and new version is possible; Phase 4 safe-publication); no alias for zero-downtime reindex; search itself is Phase 5.

**Verified:** 17 unit tests (MockTransport), 10 live tests against Compose OpenSearch 3.9 (BM25 exact id, tenant/ACL isolation, empty-ACL, kNN filtered, version replace + stale, ACL revoke, delete, cross-tenant delete, strict mapping). 9 import-linter contracts kept.

## Gate 3.8 — IngestDocument use case + failure recovery

**What:** `ekip_ingestion.pipeline.IngestDocument` (`submit`, `ingest`, `run`), `IngestionFailed(reason, stage, retryable)`, optional `ImageEnricher` hook, and `InMemoryChunkIndex` (test double with the same rules as the OpenSearch adapter).

**Flow:** `submit` stores raw bytes (key = sha256, must equal `document.content_hash`) and registers the version. `run` drives PENDING → PARSING → CHUNKING → EMBEDDING → INDEXING → INDEXED and validates the published chunk count before the final transition.

**Recovery rules (each has a test)**
- Every stage persists before the status moves; persistence is atomic/idempotent. Assets are written before the chunk set, so "chunks exist" ⇒ "assets exist".
- `run` resumes from durable state: chunks present ⇒ skip parse + chunk (embedding outage → retry does not re-parse).
- Embeddings are not stored in Postgres; the embedding cache (keyed by model, task, text hash) makes an index-outage retry free of embedding calls.
- A version left mid-stage by a dead worker is moved to FAILED("interrupted") and resumed. Assumes one worker per version; Phase 4 enforces it (Temporal workflow id).
- Failure reasons are fixed codes (`parse_error`, `no_content`, `embedding_unavailable`, `embedding_failed`, `index_unavailable`, `index_failed`, `superseded`, `storage_failed`, `invalid_document`, `unexpected_error`), never exception text, which can echo document content. `retryable` tells the caller (later: the Temporal retry policy) whether a retry can help.
- Old version cannot overwrite a newer one in the index (`superseded`, non-retryable). Re-running an INDEXED document is a no-op.

**Verified:** 15 unit tests (happy path, image chunk + asset, outages, crash resume, corrupt/empty/unsupported, stale version, submit idempotency/conflict/hash mismatch, model-version guard); live end-to-end on the Mac with Postgres + OpenSearch, with the hashing embedder and with real Gemini (768-dim): ingest → count → BM25 hit for alice, none for mallory → re-run is idempotent.

**Deferred:** image enrichment providers (OCR / vision) behind `ImageEnricher`; "invisible until validated" publication and event-driven triggers (Phase 4); HTTP upload route (needs the first real use of idempotency keys).

## Gates 4.1–4.3 — change events, repository tombstones/ACL times, ChangeHandler

**4.1 `SourceChange` (ekip_contracts.events):** kinds UPSERT / ACL_CHANGE / DELETE, lane LIVE|BACKFILL, `idempotency_key` (equal for redeliveries, different for real changes), `partition_key` (tenant + document => one Kafka partition => per-document order), strict `from_json` (only ContractError on bad input, 1 MiB cap). UPSERT does not carry bytes: the connector puts them in the object store first and names them by `content_hash`.

**4.2 Repository:** `list_versions`, `update_acl(..., as_of)`, `mark_deleted(deleted_at)`; `Document.acl_as_of` and `deleted_at` (migration 0003). Contract tests run on memory and Postgres.

**4.3 ChangeHandler (decisions by state and SOURCE time, never arrival order)**
- Same bytes => never a new version (metadata-only changes, redeliveries). A FAILED version redelivered => retried (same version).
- Older than held state => ignored. Older than a tombstone => ignored (no resurrection); strictly newer => new version.
- ACL and content have separate clocks. A backfill snapshot with newer content but an older ACL keeps the newer ACL.
- Ties favour the delete (privacy-safe side).
- Side-effect order: repository first, then index; both idempotent; duplicates repeat the index step, so a crash between them heals on retry.
- `IngestDocument` publishes with the ACL as of now, then re-reads: DELETED => remove chunks and stop (`deleted`); ACL changed => re-apply to the index. Tested with hooks injected between "chunks indexed" and "marked INDEXED".
- Version-number races: register conflict => re-read and decide again (3 attempts).

**Known limits:** a DELETE for a never-seen document leaves no tombstone (reconciliation, Gate 4.7, removes ghosts); payload erasure (raw bytes, assets, DB chunk rows) on delete is a separate retention job; cache invalidation is Phase 8.

**Verified:** 22 handler tests, 12 event tests, repository contract tests (memory + Postgres).

## Gate 4.4 — Persistent, tenant-scoped embedding reuse

**What:** `RedisEmbeddingCache` (float32, MGET / pipelined SET with TTL, 30 days), `CachingEmbedder.scoped(scope)`, `CacheStats` (hits / misses / reuse rate), `ScopableEmbedder` protocol; `IngestDocument` embeds through the per-tenant view.

**Why:** a new document version re-chunks deterministically (sections are hard boundaries), so unchanged chunks produce the same `embedding_text`; the cache turns them into hits. Measured in a test: a 10-section document with one paragraph edited costs exactly **1** new embedding on v2.

**Decisions**
- Keys are `emb:<tenant>:<model_version>:<task>:<sha256(text)>`: a model or dimension change misses automatically; the tenant segment stops one tenant seeing (or timing) another's text.
- The cache is an optimisation, never a dependency: Redis errors and corrupt entries degrade to misses (`errors` counter), a cached vector of the wrong size is a miss (poisoning guard).
- Shared interface test (`tests/contract/test_embedding_cache.py`) runs against memory and live Redis.

**Verified:** unit + contract tests; live Redis contract run on the Mac.

## Gate 4.5 — Event transport: Redpanda/Kafka, consumer, DLQ, lanes

**What:** `ekip_contracts.messaging` (ports `MessagePublisher`, `Subscription`; `Delivery`, `ChangeTopics`, `ProcessingFailure`), `ekip-messaging` (aiokafka adapters + in-memory broker with Kafka semantics), `ekip_ingestion.consumer` (`ChangePublisher`, `ChangeProcessor`, `ChangeConsumer`, `IngestTrigger`, `InlineIngestTrigger`), Redpanda in Compose (digest-pinned, dev only, localhost-bound).

**Guarantees (all tested; the broker-facing ones also against live Redpanda)**
- At-least-once: commit only after the whole batch is handled or parked in the DLQ. A failed commit, a crash before commit, or a failed DLQ publish all redeliver; the state-based ChangeHandler makes redelivery harmless.
- Per-document order: message key = tenant + document; one key's messages run sequentially, keys run concurrently (bounded by a semaphore = backpressure on the handler side).
- Lanes: live is drained before backfill is read; DELETE and ACL_CHANGE go to the live lane even if a backfill produced them (revocation never queues behind a replay).
- Poison messages: malformed or non-retryable -> DLQ immediately; retryable -> exponential backoff with jitter in place, bounded attempts, then DLQ (`<reason>_retries_exhausted`). DLQ keeps the original bytes and headers plus reason code, attempts, source topic/partition/offset, timestamp. No exception text ever reaches headers.
- Producer: idempotent, acks=all. Consumer: auto-commit off, read_committed, earliest on first start.

**Known trade-offs:** in-place retries block a partition for at most max_attempts backoffs (keeps per-document order); a batch commits as a unit. Production: managed Kafka with TLS/SASL (`client_options` pass-through), replication 3, partitions sized for consumer parallelism; DLQ replay tooling comes with the ops runbooks.

## Gate 4.6 — Durable ingestion with Temporal

**What:** `ekip-workflows` (`IngestVersionWorkflow`, `IngestActivities`, `TemporalIngestTrigger`, `build_worker`), Temporal dev server in Compose (digest-pinned, localhost-bound, in-memory state: dev only), Temporal settings.

**Ownership boundary:** Temporal decides *whether and when* a version is processed (retries, timeouts, single-flight, history). `IngestDocument` decides *what processing means* and owns the document status. The workflow carries three ids and no business state.

**Guarantees (tested live against Temporal)**
- Workflow id = `ingest:<tenant>:<document>:v<N>`; `USE_EXISTING` + `ALLOW_DUPLICATE_FAILED_ONLY`: five concurrent starts run the activity once; a start after success is accepted and does nothing; a start after a *failed* run runs again. This is the "one worker per version" assumption that the pipeline's `interrupted` recovery relies on.
- `IngestionFailed.retryable` drives Temporal's retry policy: retryable -> exponential backoff up to `max_attempts`; not retryable -> `ApplicationError(non_retryable)`, one attempt.
- Failures cross the boundary as a reason **code** only (Temporal stores failure messages in history; an exception message can echo document text). Unexpected exceptions become `unexpected_error` with no chained cause.
- The trigger starts the workflow and does not wait; Temporal unreachable -> retryable `workflow_unavailable`, so the Kafka consumer retries/DLQs rather than committing an unstarted ingest.

**Gotchas:** `result_type` is set on `get_workflow_handle`, not `result()`; `WorkflowAlreadyStartedError` lives in `temporalio.exceptions`. Workflow tests need a live server (the bundled test-server download is blocked in the sandbox), so they are gated by `RUN_INTEGRATION=1`.

## Gate 4.7 — Reconciliation against the source

**What:** `SourceConnector` port (`inventory`, `fetch`; `ekip_contracts.source`), `DocumentRepository.list_latest(tenant, source)` (memory + Postgres, contract-tested), `Reconciler` / `ReconcileReport` in `ekip_ingestion.reconcile`.

**Why:** webhooks drop, feeds expire and DLQs park events, so the event stream alone can drift from the source. A periodic "what exists now?" diff is the safety net.

**Design choice:** the reconciler changes no state. It publishes ordinary `SourceChange` events (backfill lane; DELETE and ACL_CHANGE are routed to the live lane), so ordering, idempotency and tombstone rules stay in `ChangeHandler` only. Upserts store the fetched bytes in the object store first, because the handler requires content to exist before it registers a version.

**Rules (all tested)**
- Missing -> UPSERT; `source_version` differs -> UPSERT (metadata-only changes become DUPLICATE downstream); ACL differs and the listing is not older than our ACL -> ACL_CHANGE (no re-embedding); absent from the listing -> DELETE.
- Tombstone newer than the source copy -> not revived (fail closed); source copy newer than the tombstone -> recreated.
- "Absent" means "delete", so: an incomplete listing raises `source_unavailable` before any event is sent, and a mass-delete guard (>30% of live documents and at least 10) withholds the deletes and reports them.
- Fetch failures and vanished items are counted and skipped, never turned into deletes. Tenant and source scoping verified.
- Convergence test: after missed edit/delete/permission/create events the system reaches the source's state, and the next run emits nothing.

**Known limits:** one tenant+source is diffed in memory (shard by prefix beyond millions); an UPSERT stamps the ACL with the event time, so a listing's `acl_as_of` only matters for ACL_CHANGE events; scheduling the reconciler (Temporal schedule) comes with the worker service.

**Bug found at the Phase 4 close-out (Gate 4.5 code):** `KafkaPublisher.close()` skipped `stop()` when `start()` had failed (broker unreachable), leaking the producer's client and tasks. CI runs pytest with `-W error`, which turned the resulting `ResourceWarning` into a failure in an unrelated test. `close()` now always stops the producer exactly once.


# Phase 5 - Retrieval engine

## Gates 5.1-5.2 - Identity in the query
**What:** `principal` + `groups` token claims -> `SecurityContext` -> `SearchScope`; `ChunkSearcher` port; OpenSearch BM25 (`multi_match`) and kNN with the tenant + ACL filter *inside* the query (Lucene engine filter in the knn clause), routing by tenant.
**Why:** post-filtering top-k leaks via ranking and starves callers (the nearest vectors may all be restricted). Pushdown returns the best *readable* hits. No identity -> `match_none` (fail closed). Groups are capped (500); malformed entries are dropped.
**Gotcha:** BM25 term statistics in real OpenSearch are shard-wide (a tiny statistics side channel across tenants sharing a shard); the in-memory twin uses visible rows only. Accepted for now; dedicated index per large tenant removes it.

## Gate 5.3 - Hybrid retriever
Parallel BM25 + kNN fused with RRF (k=60, rank based, deterministic tie-break). Only `SearchIndexError`/`EmbeddingError` count as degradation; any other exception propagates. Query embeddings cached per tenant.

## Gate 5.4 - Reranker port
`HttpReranker` (Cohere/Jina/Voyage-style `/rerank`): https only, off by default, errors carry only the status. **Not verified against a live provider** (needs a key and an egress allowlist entry). `TermCoverageReranker` is a labelled baseline.

## Gate 5.5 - Context assembly and citations
Dedupe (hash or normalised text), per-document cap, token budget (chars/4 until a real tokenizer is injected), truncation at word boundaries, numbered citations. `render()` escapes markup so retrieved text cannot close its `<source>` element. `check_citations` separates valid from invented `[n]`. Note: `array[1]` in an answer is indistinguishable from a citation marker.

## Gate 5.6 - API wiring
`/api/v1/search` and the chat stub now use `SearchScope.from_context`. Backend switch `RETRIEVAL_BACKEND=memory|opensearch` (default memory: seeded demo corpus with ACLs, so smoke needs no ingestion). Keycloak realm gained `principal` and `groups` mappers and groups eng/ops (carol is in both). **Finding:** kNN always returns neighbours, so an unmatched query still returns weak hits from the caller's *own* tenant; a relevance floor is future work.

## Gate 5.7 - Evaluation
Offline (hashing embedder): hybrid recall 1.00 vs BM25 0.97 vs vector 0.86. Live (OpenSearch + Gemini): vector 1.00/MRR 1.00, hybrid 1.00/0.92, BM25 0.97/0.89, hybrid+term-coverage 0.97/0.86; 0 leaks everywhere. Reading: with a real embedding model, vector wins on this small paraphrase-heavy set; the baseline reranker hurts, so none is default. Live latency (~550 ms) is the uncached Gemini query embedding. Do not generalise from 36 queries.
