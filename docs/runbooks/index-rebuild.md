# Index rebuild and zero-downtime re-index

Use when: the embedding model or dimensions change, the OpenSearch mapping changes incompatibly (`MAPPING_VERSION`), the index is corrupt, or `EkipRetrievalDegraded` fires because of index problems.

## Detection
Retrieval latency or error alerts; `retrieval_degraded` reasons in logs; planned model/mapping change.

## Diagnosis
1. The chunk index name is derived from the embedder's model version: `ekip-chunks-<model-version-slug>`. A new embedding model therefore gets a **new index**; the old one is untouched while the new one is built.
2. `GET _cluster/health` and `GET _cat/indices/ekip-chunks-*?v` show state, size and doc counts. Compare chunk counts with the document registry in Postgres (source of truth for what should be indexed).

## Action (planned change, no downtime)
1. Capacity first: the new index needs space for a full second copy plus headroom (see `docs/capacity-model.md`; measure, do not guess).
2. Deploy the new embedding configuration to a **staging** instance pointing at the same OpenSearch; it creates the new index.
3. Backfill through the **backfill lane** (reconcile or re-publish all documents). The live lane keeps serving; live changes must be written to both indices until cut-over, or re-reconciled after.
4. Evaluate the new index offline and live: `uv run --no-sync python scripts/eval_retrieval.py --live` and `uv run --no-sync python scripts/eval_gate.py`. Recall must not regress; the leak count must be 0.
5. Cut over: roll the production API/workers to the new embedding configuration. Keep the old index for the rollback window (agreed with the owner, default 7 days).
6. Update `ops/model-registry.yaml` models section and the cache generation (the corpus changed: see `cache-invalidation.md`).

## Action (damaged index)
Create the index from the mapping, then reconcile every tenant. Do not restore from a snapshot taken during the damage window without checking it (see `backup-restore-dr.md`).

## Verification
- Doc counts match the registry; the eval gate passes; a sample of known questions returns cited answers; `ekip_retrieval_degraded_total` flat.
- Revoked or deleted documents are absent in the new index (spot-check three).

## Escalation
If the old index is gone and the new one is incomplete, page the search owner. Missing content shows up as "insufficient evidence" answers (the answer path refuses rather than guessing); tell users which tenants are affected.
