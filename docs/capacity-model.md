# Capacity model

Reproduces the arithmetic of masterplan section 6 so it can be re-run with real inputs
(`ekip_observability.capacity`, tested in `tests/unit/test_observability_capacity.py`).

> **One million documents is a storage and indexing requirement, not enough information to size
> the platform.** Everything marked *assumption* below must be replaced by a measurement from a
> representative corpus before it is used to buy hardware.

## Inputs

| Input | Planning value | Status |
|---|---:|---|
| Initial documents | 1,000,000 | requirement |
| Average chunks per document | 20 | assumption - measure with the real chunker |
| Embedding dimensions | 1,024 | assumption - the dev stack uses 768 (`EMBEDDING_DIMENSIONS`) |
| Bytes per dimension | 4 (FP32) | assumption - quantization changes it |
| Daily changed documents | 10,000 | assumption |
| Concurrent users, tokens per request, latency targets | not yet known | **required inputs** |

## Cell-by-cell arithmetic

| Quantity | Formula | Value |
|---|---|---:|
| Total chunks | 1,000,000 x 20 | 20,000,000 |
| Raw vector bytes | 20,000,000 x 1,024 x 4 | 81,920,000,000 B = **81.92 GB** (decimal GB) |
| Changed chunks / day (upper bound) | 10,000 x 20 | 200,000 |
| Raw vectors incl. 1 replica | 81.92 x 2 | 163.84 GB |
| Backfill time | total chunks / measured embeddings per second | e.g. 20,000,000 / 1,000 per s = 5.6 h *(rate is illustrative)* |

In code: `CorpusAssumptions().raw_vector_gb == 81.92`, `replicated_vector_gb(a, replicas=1)`,
`backfill_hours(chunks, measured_rate)`.

## What 81.92 GB does *not* include

ANN graph structures (HNSW links add per-vector overhead), BM25 postings and stored fields, chunk
text and metadata, replicas, segment merge overhead, free-disk watermarks, and a **second full index
during a re-index** (`docs/runbooks/index-rebuild.md`). Do not state that an 82 GB disk is sufficient.

## What to measure (Phase 11 benchmarks)

1. Chunks per document and text bytes per chunk on a representative sample (>= 10,000 documents).
2. Index size on disk after ingesting that sample; extrapolate linearly, then verify at 10x.
3. Initial backfill throughput (documents/min, chunks/min, embeddings/s) and the bottleneck stage.
4. Incremental lag under steady change load, with a backfill running at the same time (the live lane must not starve).
5. Query latency (P50/P95/P99) versus concurrency, with and without ACL filters, with warm and cold caches.
6. Embedding-provider rate limits and cost per million tokens.
7. Replica recovery time and full re-index duration.
8. Tokens per chat request (context size) and cost per answered request; feed `ekip_llm_cost_usd_total`.

Use `scripts/load_test.py` for item 5 and the `ekip_*` metrics (dashboards in `ops/grafana/`) for
the rest. Record results next to the date, hardware and dataset version; a number without those is
not reusable.

## Decisions this model does not make

Node counts, instance types, shard counts and GPU needs. Those follow from the measurements above
and are deployment-time decisions (infrastructure is covered separately).
