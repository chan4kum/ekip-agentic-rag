# Cache flush and invalidation

Alerts: `EkipCacheBlockedEntries`. Invariant I-9: caches never bypass authorization; keys include tenant and permission-scope version, and revocation invalidates entries.

## Detection
- `ekip_cache_events_total{event="blocked"}` or `{event="stale"}` rising: entries were refused because the permission scope or corpus version moved.
- Reports of an answer citing a deleted document or revealing something a user lost access to (treat that as `permission-revocation-incident.md` first, this runbook second).

## Diagnosis
1. Which layer? The metric label `layer` is one of response, retrieval, embedding, semantic.
2. Did invalidation run? After a permission change, deletion or re-index the scope/corpus generation must advance; entries from the old generation are unreachable by key. A stale hit means a key was built without the current generation - a bug.
3. Redis health: `docker compose exec redis redis-cli ping`, memory and eviction (`INFO memory`).

## Action
- Normal case: nothing to flush. Advance the affected tenant's generation (the mechanism lives in `ekip_caching`, see ADR-010); old entries expire by TTL.
- Emergency purge of one layer: delete keys with that layer's prefix using `SCAN` + `UNLINK` in batches. **Never `FLUSHALL`/`FLUSHDB`**: Redis also holds rate-limit counters and idempotency records; wiping them re-opens replay and abuse windows.
- Disable a cache layer by configuration if its invalidation is suspect; correctness is worth the latency cost.

## Verification
- Ask the offending question as the affected user: the answer reflects current permissions and documents; `blocked`/`stale` counters stop growing.
- Hit ratio recovers over the next hour (it will dip first - that is expected).

## Escalation
Any cached answer served to a user who is not entitled to it is a security incident: page security on-call immediately and disable all answer-level caching until the cause is found.
