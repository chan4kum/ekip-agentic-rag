"""Measure retrieval quality and access safety on a versioned dataset.

Why: "it seems to work" is not evidence. Every retrieval change (chunking, embedding model,
fusion, reranker) must be compared on the same questions, and the security check - did anyone
get a passage they may not read? - must stay at exactly zero.

Dataset (JSON, versioned): documents with tenant + ACL, and queries with the caller's identity and
the passages that answer them. A query whose answer exists only in documents the caller may not
read has an EMPTY `relevant` list: it contributes to the leak check, not to recall/MRR/NDCG.
"""

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ekip_contracts.documents import Chunk, ChunkEmbedding, Document, DocumentAcl
from ekip_contracts.embedding import Embedder, EmbeddingTask
from ekip_contracts.retrieval import Retriever, SearchScope
from ekip_contracts.search_index import ChunkIndex, IndexEntry


@dataclass(frozen=True)
class EvalPassage:
    key: str  # unique across the dataset, e.g. "mx-torque#0"
    section: str
    text: str


@dataclass(frozen=True)
class EvalDocument:
    id: str
    tenant: str
    title: str
    users: tuple[str, ...]
    groups: tuple[str, ...]
    passages: tuple[EvalPassage, ...]


@dataclass(frozen=True)
class EvalQuery:
    id: str
    text: str
    kind: str  # exact | paraphrase | multi | restricted | cross_tenant
    scope: SearchScope
    relevant: frozenset[str]  # passage keys; empty = nothing readable answers this


@dataclass(frozen=True)
class EvalDataset:
    version: str
    description: str
    documents: tuple[EvalDocument, ...]
    queries: tuple[EvalQuery, ...]


def load_dataset(path: Path) -> EvalDataset:
    raw = json.loads(path.read_text())
    docs = tuple(
        EvalDocument(
            id=d["id"],
            tenant=d["tenant"],
            title=d["title"],
            users=tuple(d.get("users", ())),
            groups=tuple(d.get("groups", ())),
            passages=tuple(
                EvalPassage(f"{d['id']}#{i}", p["section"], p["text"])
                for i, p in enumerate(d["passages"])
            ),
        )
        for d in raw["documents"]
    )
    queries = tuple(
        EvalQuery(
            id=q["id"],
            text=q["text"],
            kind=q["kind"],
            scope=SearchScope(
                q["tenant"], user=q.get("user", ""), groups=frozenset(q.get("groups", ()))
            ),
            relevant=frozenset(q.get("relevant", ())),
        )
        for q in raw["queries"]
    )
    keys = {p.key for d in docs for p in d.passages}
    for q in queries:
        if not q.relevant <= keys:
            raise ValueError(f"query {q.id} names unknown passages {sorted(q.relevant - keys)}")
    return EvalDataset(raw["version"], raw["description"], docs, queries)


async def ingest_dataset(
    index: ChunkIndex, embedder: Embedder, dataset: EvalDataset
) -> dict[str, str]:
    """Index every document; returns chunk_id -> passage key."""
    chunk_to_key: dict[str, str] = {}
    for d in dataset.documents:
        doc = Document.create(
            tenant_id=d.tenant,
            source="eval",
            source_document_id=d.id,
            version=1,
            source_version="v1",
            content=d.id.encode(),
            title=d.title,
            mime_type="text/plain",
            source_uri=f"https://eval.test/{d.tenant}/{d.id}",
            acl=DocumentAcl("eval-acl", frozenset(d.users), frozenset(d.groups)),
            last_modified=datetime(2026, 10, 19, tzinfo=UTC),
        )
        chunks = [
            Chunk.create(document=doc, ordinal=i, content=p.text, section_path=(p.section,))
            for i, p in enumerate(d.passages)
        ]
        vectors = await embedder.embed([c.content for c in chunks], task=EmbeddingTask.DOCUMENT)
        await index.replace_document(
            doc,
            [
                IndexEntry(c, ChunkEmbedding(c.chunk_id, c.content_hash, embedder.model_version, v))
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
        for c, p in zip(chunks, d.passages, strict=True):
            chunk_to_key[c.chunk_id] = p.key
    return chunk_to_key


# ---- metrics (binary relevance) ---------------------------------------------------------------


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    for i, key in enumerate(ranked[:k], start=1):
        if key in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1) for i, key in enumerate(ranked[:k], start=1) if key in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


@dataclass(frozen=True)
class Metrics:
    queries: int  # queries that have at least one readable relevant passage
    recall: float
    mrr: float
    ndcg: float
    hit_rate: float  # share of queries with >=1 relevant passage in the top k


@dataclass(frozen=True)
class EvalReport:
    name: str
    k: int
    overall: Metrics
    by_kind: Mapping[str, Metrics]
    leaks: tuple[str, ...]  # "query_id -> passage key" for every unreadable passage returned
    restricted_queries: int
    mean_latency_ms: float = 0.0
    failures: tuple[str, ...] = field(default=())


def _aggregate(rows: Sequence[tuple[float, float, float, float]]) -> Metrics:
    n = len(rows)
    if n == 0:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0)
    return Metrics(
        n,
        sum(r[0] for r in rows) / n,
        sum(r[1] for r in rows) / n,
        sum(r[2] for r in rows) / n,
        sum(r[3] for r in rows) / n,
    )


def readable_keys(dataset: EvalDataset, scope: SearchScope) -> frozenset[str]:
    """Leak-check ground truth: computed from the dataset, not from the system under test."""
    keys: set[str] = set()
    for d in dataset.documents:
        if d.tenant != scope.tenant_id:
            continue
        if (scope.user and scope.user in d.users) or (set(d.groups) & scope.groups):
            keys.update(p.key for p in d.passages)
    return frozenset(keys)


async def evaluate(
    name: str,
    retriever: Retriever,
    dataset: EvalDataset,
    chunk_to_key: Mapping[str, str],
    *,
    k: int = 5,
    clock_ms: Callable[[], float] | None = None,
) -> EvalReport:
    now = clock_ms or (lambda: time.perf_counter() * 1000)
    rows: list[tuple[float, float, float, float]] = []
    by_kind_rows: dict[str, list[tuple[float, float, float, float]]] = {}
    leaks: list[str] = []
    failures: list[str] = []
    restricted = 0
    elapsed = 0.0
    for q in dataset.queries:
        started = now()
        try:
            hits = await retriever.search(q.text, scope=q.scope, limit=k)
        except Exception as exc:  # a failing query is reported, never silently skipped
            failures.append(f"{q.id}: {type(exc).__name__}")
            continue
        elapsed += now() - started
        ranked = [chunk_to_key.get(h.chunk_id, f"?{h.chunk_id}") for h in hits]
        allowed = readable_keys(dataset, q.scope)
        leaks.extend(f"{q.id} -> {key}" for key in ranked if key not in allowed)
        if not q.relevant:
            restricted += 1
            continue
        row = (
            recall_at_k(ranked, q.relevant, k),
            reciprocal_rank(ranked, q.relevant, k),
            ndcg_at_k(ranked, q.relevant, k),
            1.0 if set(ranked[:k]) & q.relevant else 0.0,
        )
        rows.append(row)
        by_kind_rows.setdefault(q.kind, []).append(row)
    answered = len(dataset.queries) - len(failures)
    return EvalReport(
        name=name,
        k=k,
        overall=_aggregate(rows),
        by_kind={kind: _aggregate(r) for kind, r in sorted(by_kind_rows.items())},
        leaks=tuple(leaks),
        restricted_queries=restricted,
        mean_latency_ms=elapsed / answered if answered else 0.0,
        failures=tuple(failures),
    )


def format_reports(reports: Sequence[EvalReport]) -> str:
    """A compact comparison table (Markdown)."""
    if not reports:
        return ""
    k = reports[0].k
    head = (
        f"| variant | queries | Recall@{k} | MRR | NDCG@{k} | hit-rate "
        "| leaks | failed | ms/query |"
    )
    lines = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        m = r.overall
        lines.append(
            f"| {r.name} | {m.queries} | {m.recall:.3f} | {m.mrr:.3f} | {m.ndcg:.3f} | "
            f"{m.hit_rate:.3f} | {len(r.leaks)} | {len(r.failures)} | {r.mean_latency_ms:.1f} |"
        )
    return "\n".join(lines)
