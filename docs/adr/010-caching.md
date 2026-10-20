# ADR-010: Caching with bound keys, access re-checks and instant invalidation

Status: accepted (2026-10-20). Applies to Phase 8 (`packages/caching`, `ekip_contracts.cache`).

## Context
Repeated questions and repeated retrievals cost latency and money. A cache is also the easiest
place to break the platform's security model: it replays work done for someone else, possibly
before a document was deleted or a permission was revoked. The rule for this phase is therefore:
**a cache may make answers faster, never different, and never visible to the wrong person.**

Five caches exist in the architecture; they solve different problems and are not interchangeable.

| Layer | Where | Status |
|---|---|---|
| Embedding cache | Redis (`RedisEmbeddingCache`, Phase 5) | done |
| Retrieval cache | Redis via `RetrievalCache` | this ADR |
| Response cache (exact match) | Redis via `ResponseCache` | this ADR |
| Semantic response cache | Redis + vector similarity | **not built** (see below) |
| LLM KV / prompt cache | model provider or vLLM prefix caching | configured at the model, not in this package |

Redis does not provide transformer KV caching; that lives in the inference engine or provider.

## Decision
1. **Keys bind everything that shapes an answer.**
   `sha256(tenant, caller scope fingerprint, authorization_scope_version, normalised query,
   corpus generation, retrieval_config_version, prompt_version, model_version,
   output_policy_version)` with length-prefixed fields and a domain-separation header. The raw
   query never appears in a Redis key. The caller fingerprint covers user AND sorted groups, so
   two identities never share an entry (no "same groups, same answer" shortcut: a user-specific
   ACL entry would make that unsafe).
2. **Every hit is re-checked.** The cache stores the ids of every document whose text reached the
   prompt (`source_document_ids`, cited or not). On a hit, `VisibilityChecker.all_visible(scope,
   ids)` must return True. False drops the entry (`stale`). Any exception or timeout is treated
   as a miss (`blocked`): **fail closed**. The key is a performance device; this check is the
   security control. The integrator backs the port with the same index/ACL query retrieval uses.
3. **Invalidation is immediate for security, targeted for freshness.**
   - Document delete and ACL change: bump the tenant's *corpus generation* first. Every earlier
     entry of that tenant stops matching at once; then purge entries tagged with the document to
     free memory. A failed bump raises `CacheInvalidationError` so the workflow retries.
   - New version published: purge entries tagged with that document (no stale citation). A
     bump is optional (`bump_on_publish`). Without it, a query that would merely *newly rank* a
     fresh document keeps its old cached result until TTL; this is a freshness trade-off only.
   - The generation starts at the current time in ms, not 0, so a counter lost to eviction or a
     flush cannot repeat an old value. Run the cache Redis with `volatile-lru` (counters have no
     TTL and are never evicted).
4. **Single-flight.** N identical concurrent misses (same full key, hence same caller) run one
   computation per process. A cancelled leader hands over to a waiter instead of failing them.
   Cross-process duplication is accepted rather than adding a distributed lock.
5. **The cache is never required.** Store errors, decode errors, oversize values and bound-key
   mismatches all become "compute normally" and are counted. Entries carry an envelope
   (version, layer, tenant|caller binding); bytes found under the wrong binding are ignored.
6. **What is not cached.** Callers without an ACL identity; answers with no source documents
   (refusals / "no evidence" must not mask a document ingested later); empty retrieval results;
   values over `max_bytes`; entries citing more than `max_tags` documents (they could not be
   purged per document).
7. **Metrics are a one-method port** (`CacheMetrics.record(cache, outcome)`, outcomes: hit, miss,
   stale, blocked, corrupt, error, skipped); the observability package supplies the Prometheus
   implementation.

## Semantic response cache: deferred, with entry criteria
Serving an answer for a *similar* question adds risk the exact cache does not have. It stays out
until a versioned evaluation set shows all of the following:
- **Equivalence:** pairs of paraphrases that must share an answer, and near-miss pairs that must
  not ("rev 6" vs "rev 7", "Is X allowed" vs "Is X not allowed"); measured false-hit rate is
  zero on the near-miss set at the chosen similarity threshold.
- **Citation freshness:** after a document update, delete or ACL change, no semantic hit may
  return superseded content (same generation + tag rules and re-check as above).
- **Authorization safety:** the similarity search is filtered by tenant and caller fingerprint
  *inside* the query, and the visibility re-check still runs on every hit.
Until then similar questions are served by the embedding and retrieval caches plus the model's
provider-side prompt cache.

## Consequences
- A hit costs one store read plus one visibility query. The visibility query is the price of
  correctness and must be cheap (id lookup with the caller's ACL filter), or the cache gains
  little. Measure it before turning the response cache on.
- Any change to prompts, model, retrieval settings or output policy must change the matching
  version string, otherwise old answers keep being served. These strings belong in the
  prompt/model registry, not in code constants.
- Deleting or re-permissioning one document retires the whole tenant's cached answers. That is
  deliberately blunt: it is correct without needing a reverse index of who saw what. If churn
  makes it expensive, move to per-document tags on the *read* side first and measure.

## Benchmark honesty
`scripts/bench_cache.py` times key building and the in-memory hit path in one process with a
fake visibility checker. It exists to catch regressions in our own code (microseconds). It says
nothing about production: there Redis round trips, serialization of real answers, the real
visibility query and network jitter dominate. No latency or hit-rate claim may be made for the
platform until Phase 11 measures it against a representative corpus and query log.

## Tests that guard this ADR
Cross-tenant and cross-identity isolation, revocation and deletion, visibility failure, version
changes, TTL, stampede, corrupted / oversized / mis-bound values, unreachable Redis, generation
loss, invalidation failure (`tests/unit/test_cache_*.py`), and one store contract run against
memory and Redis (`tests/contract/test_cache_store.py`, Redis when `RUN_INTEGRATION=1`).
