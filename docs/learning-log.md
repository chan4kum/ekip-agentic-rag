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
