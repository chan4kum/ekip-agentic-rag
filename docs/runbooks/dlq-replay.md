# Dead-letter queue: inspect and replay

Alerts: `EkipDeadLetterQueueNotEmpty`, `EkipDeadLetterQueueGrowing`.

## Detection
`ekip_ingest_dlq_depth > 0` for 30 min, or above 100. The topic is `<prefix>.changes.dlq` (default `ekip.changes.dlq`).

## Diagnosis
Each parked message keeps the original key and value and gains headers: `dlq-reason` (a fixed code, never exception text), `dlq-attempts`, `dlq-source` (`topic:partition:offset`), `dlq-at`.
1. Read a sample: `docker compose exec redpanda rpk topic consume ekip.changes.dlq -n 20 -f '%k %h\n'` (values can contain document metadata; do not paste them into tickets).
2. Group by `dlq-reason`. Non-retryable reasons (malformed event, unsupported type) need a code or data fix. Retryable reasons that exhausted attempts (dependency outage) usually only need the dependency back.

## Action
There is **no automated replay tool yet** (known gap). Use one of:
1. **Preferred - reconcile.** Fix the root cause, then run the reconciler for the tenant. It lists the source of truth and republishes whatever differs, so it also recovers items missed for other reasons. It is idempotent.
2. **Targeted re-publish.** After the fix, re-produce the specific DLQ values to the *live* topic: `rpk topic consume ... -f '%k\t%v\n'` then `rpk topic produce ekip.changes.live -k <key>`. Only for a handful of messages; re-check each is still wanted (a deleted-then-replayed upsert is stopped by tombstones, but do not rely on it).
3. **Discard.** Only for messages proven obsolete; record the decision and the reason in the incident ticket. Never delete the topic.

## Verification
- DLQ depth returns to 0 and stays there; the replayed documents are searchable (or correctly absent if deleted).
- No new `dlq-reason` of the same code appears in the next hour.

## Escalation
Same reason recurring after the fix, or revocations among the parked messages: page the data-platform on-call and open `permission-revocation-incident.md`.
