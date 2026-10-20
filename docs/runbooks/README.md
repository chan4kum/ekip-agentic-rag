# Runbooks

Operational procedures for EKIP. Every runbook has the same five sections: **Detection**,
**Diagnosis**, **Action**, **Verification**, **Escalation**. Alerts in `ops/prometheus/alerts.yaml`
point at these files (a test checks the links).

| Runbook | When |
|---|---|
| [ingestion-backlog.md](ingestion-backlog.md) | Freshness or consumer-lag alerts |
| [dlq-replay.md](dlq-replay.md) | Messages sitting in the dead-letter topic |
| [index-rebuild.md](index-rebuild.md) | New embedding model, mapping change, corrupt or degraded index |
| [cache-invalidation.md](cache-invalidation.md) | Stale or blocked cache entries, permission or corpus change |
| [llm-provider-outage.md](llm-provider-outage.md) | Model provider errors, latency or spend alerts |
| [key-rotation.md](key-rotation.md) | Scheduled or emergency rotation of API keys and secrets |
| [backup-restore-dr.md](backup-restore-dr.md) | Backups, restore, disaster-recovery drill |
| [permission-revocation-incident.md](permission-revocation-incident.md) | Access removed at the source is still effective in EKIP |

Conventions: commands assume the local Compose stack (`docker compose ...`). Production commands
differ by environment and are added to each runbook when the environment is built. Never paste
tokens, keys or document content into tickets or chat; use request ids and trace ids.
Placeholders look like `<tenant>`.
