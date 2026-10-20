# Backup, restore and disaster-recovery drill

Scope: PostgreSQL (document registry, conversations, idempotency, workflow state), OpenSearch (derived index), object storage (raw documents), Redis (ephemeral), Kafka/Redpanda (transport). **Source of truth:** raw documents in object storage plus the document registry. The search index is *derived* and can be rebuilt; losing it costs time, not data.

Environment-specific backup mechanics (managed snapshots, retention, cross-region copies) are filled in when the production environment is built. Targets below are placeholders to be agreed: **RPO** and **RTO** per component must be written down before the first drill.

## Detection
Backup job failed or is older than its schedule; restore test overdue; region or storage failure.

## Diagnosis
1. Which components are lost? Postgres and object storage lost = data incident. OpenSearch or Redis only = rebuild.
2. Last verified-good backup: its time is your actual RPO.

## Action - restore order
1. Object storage (raw documents) and Postgres first, to a consistent point in time. Check that registry rows reference objects that exist.
2. Bring up the identity provider and API in **read-only/degraded** mode.
3. Rebuild OpenSearch from the registry by reconciling every tenant (`index-rebuild.md`); restore an index snapshot only if it is newer than the last known-good change and verified.
4. Flush answer caches by advancing generations (`cache-invalidation.md`): restored data may be older than cached answers.
5. Replay or re-reconcile changes made after the restore point; permission changes first (live lane).

## Drill checklist (at least twice a year, and after major changes)
- [ ] Pick a drill date and scope; tell stakeholders.
- [ ] Restore Postgres into an isolated environment from the backup, not from the live database.
- [ ] Restore raw documents; verify checksums of a random sample (registry stores `content_hash`).
- [ ] Rebuild the index for one tenant; time it; compare doc counts.
- [ ] Run `scripts/eval_gate.py` against the restored stack; leak count must be 0.
- [ ] Verify a revoked document stays invisible after the restore.
- [ ] Record measured RPO and RTO, gaps found, and owners for fixes.
- [ ] Destroy the drill environment and its data.

## Verification
Smoke test passes, eval gate passes, doc counts match the registry, permissions spot-check passes.

## Escalation
Backups older than the agreed RPO or a failed restore test: page the platform owner the same day; no deployments until a good backup exists.
