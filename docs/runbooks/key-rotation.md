# Key and secret rotation

Applies to: LLM API key, embedding key, Tavily/search keys, database and Redis passwords, Keycloak client secrets, the metrics scrape token, Langfuse keys. Invariant I-8: secrets live only in the environment / secret manager - never in code, prompts, logs, traces or git.

## Detection
Scheduled (calendar, at most 90 days) or emergency: a key appeared in a log, ticket, screenshot, commit or a trace; an ex-employee had access; the provider reports abuse.

## Diagnosis (emergency)
1. Which secret, where exposed, since when? Search git history and logs for the *name* of the variable first; for the value use the provider's usage dashboard rather than pasting the value anywhere.
2. Check provider usage for calls you do not recognise.

## Action
1. Create the **new** key in the provider console (do not revoke yet).
2. Put it in the secret store / `.env` (local). Never in the command line history: edit the file or use the secret manager UI.
3. Restart the consumers of that secret (API, workers). For local Compose: `docker compose up -d api`.
4. Confirm traffic uses the new key (provider usage shows it; `ekip_llm_errors_total{kind="auth"}` stays flat).
5. **Revoke the old key.** Emergency: revoke first, accept a short outage.
6. Database/Redis passwords: change in the service, then in the secret store, then restart dependants - in that order, in a maintenance window.

## Verification
`/ready` is green, `scripts/smoke.sh` passes, a chat request answers with citations, no `auth` errors for 30 min. Old key returns 401/403 when tried against the provider.

## Escalation
Exposure outside the company or evidence of use by someone else: security on-call, treat as an incident, review the audit log for the exposure window.
