# Ingestion backlog (freshness or consumer lag)

Alerts: `EkipIngestionFreshnessP95High`, `EkipConsumerLagHigh`, `EkipIngestFailureRatioHigh`.

## Detection
- Dashboard `EKIP / Ingestion`: freshness P95 above 5 min, consumer lag growing, documents per minute falling.
- Users report that a changed document still shows its old content.

## Diagnosis
1. Is the lag on the **live** lane or the **backfill** lane? Live (`ekip.changes.live`) carries user-visible changes, deletions and permission revocations; backfill (`ekip.changes.backfill`) carries bulk replays. A backfill must never delay the live lane - if it does, that is a bug.
2. Check consumer health: `docker compose logs --since 15m api | grep -i consumer`; are workers alive and committing offsets?
3. Failure ratio rising? Group failures by reason in the logs (`dlq-reason` header, or `reason=` log field). Typical causes: parser errors on one file type, embedding provider rate limits, OpenSearch rejecting bulk writes (disk watermark, mapping mismatch).
4. Dependencies: `GET /ready` shows which probe fails (database, Redis, OpenSearch, broker).

## Action
- Provider rate-limited: reduce embedding concurrency or batch size; wait for quota; do not retry in a tight loop.
- OpenSearch unhealthy: free disk or add capacity first, then let the consumer catch up. Do not delete indices to "make room" without the index-rebuild runbook.
- One poison document: it should already be parked in the DLQ (see `dlq-replay.md`); the stream continues.
- Large backfill running: pause or throttle the backfill consumer, keep the live consumer running.
- Consumer stuck: restart the consumer process. Delivery is at-least-once and handlers are idempotent, so a restart is safe.

## Verification
- Lag falls to near zero; freshness P95 under 5 min for 30 min.
- Pick a recently changed document, search for new text: it is found and the old text is not.
- DLQ depth is not growing.

## Escalation
Page the data-platform on-call if live-lane lag exceeds 30 min or any permission change is delayed (then also run `permission-revocation-incident.md`).
