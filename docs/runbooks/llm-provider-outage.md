# LLM provider outage, latency or spend anomaly

Alerts: `EkipApiErrorBudgetFastBurn/SlowBurn`, `EkipSimpleRagLatencyP95High`, `EkipFirstTokenP95High`, `EkipLlmErrorRateHigh`, `EkipLlmSpendSpike`, `EkipAgentBudgetTruncationsHigh`, `EkipAgentFailureRatioHigh`, `EkipRateLimitRejectionsHigh`.

## Detection
Dashboard `EKIP / LLM`: provider error ratio, P95 latency by model, token and spend panels. `EKIP / Agents`: failures and budget truncations.

## Diagnosis
1. Provider side or ours? Compare `ekip_llm_errors_total` by `kind` (`timeout`, `rate_limited`, `auth`, `server`). `auth` = key problem (see `key-rotation.md`); `rate_limited` = quota or our own concurrency; `server`/`timeout` = provider incident (check its status page).
2. Is retrieval healthy? If `ekip_retrieval_duration_seconds` is also high the problem is not the model.
3. Spend spike: which `model` label? A looping agent shows as truncations plus tokens; a traffic spike shows in request rate. Look for one tenant or client via request ids in logs (never as metric labels).

## Action
- The answer path fails closed: when the model is unavailable users get an explicit error or "insufficient evidence", never an ungrounded guess. Do not weaken that to hide an outage.
- Switch model/provider by configuration (`LLM_BASE_URL`, `LLM_MODEL`; any OpenAI-compatible endpoint works). Run `scripts/eval_gate.py --suites retrieval,agent` against the fallback before relying on it; record the change in `ops/model-registry.yaml`.
- Rate limited: lower concurrency, tighten per-user limits; do not add blind retries.
- Runaway spend: lower the agent budgets (steps, model calls, tokens) or switch back to the single-step pipeline (see the agent settings in `.env.example`), and rotate the key if abuse is suspected.

## Verification
- Error ratio and P95 back under SLO for 30 min; `scripts/smoke.sh` passes; spend rate normal.

## Escalation
Page the platform on-call when the 99.9% error budget burn alert fires. Contact the provider's support with request ids and timestamps, not content.
