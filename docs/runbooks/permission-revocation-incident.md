# Permission revocation incident

Situation: access was removed or a document was deleted at the source (SharePoint, Drive, ...) but EKIP still returns the content, or a user sees content they should not. Invariants I-4, I-5, I-9: this is a security incident until proven otherwise.

Alert: `EkipPermissionPropagationSlow` (P95 above 60 s) - or a report from a user or owner.

## Detection
- Report: "user X can still get answers from document Y".
- `ekip_acl_propagation_seconds` high; DLQ or live-lane lag growing.

## Diagnosis
1. Reproduce as the affected user with a test account (never ask for someone's token). Note request id and trace id.
2. Was the revocation event received? Look in the live topic and the DLQ for the document. Deletions and revocations always use the **live** lane.
3. Did the index change? ACL fields (`acl_version`) and the `is_deleted` state on the chunks. Was the event processed but the cache serving an old answer (`cache-invalidation.md`)?
4. Was the change never emitted? Then the connector missed it - the periodic reconciliation should catch it; run it now.

## Action
1. **Contain first.** Block the document: mark it deleted/hidden in the registry and the index (fail-closed ACL), or temporarily remove the affected group from the tenant's allowed scope. Containment must not wait for root cause.
2. Advance the cache generation for the tenant so no cached answer survives.
3. Re-run the reconciler for the tenant so the index matches the source.
4. Find the cause (missed webhook, consumer stalled, bug in ACL mapping) and fix; add a regression test (`tests/security`).
5. Review the audit log for the exposure window: who retrieved the document, when. Hand the list to the data owner and security.

## Verification
- The test account no longer retrieves the content, directly or via cache, in search and in chat.
- Propagation delay back under 60 s P95; no other revocation is pending in the DLQ.

## Escalation
Immediately: security on-call and the data owner. If content left the company, follow the company incident process; this runbook only covers containment and recovery.
