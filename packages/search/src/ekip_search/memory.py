"""In-memory ChunkIndex for tests and local development (same rules as the OpenSearch adapter)."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ekip_contracts.documents import ContractError, Document, DocumentAcl
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_contracts.search_index import IndexEntry, StaleIndexVersionError

MAX_SEARCH_LIMIT = 200
_K1, _B = 1.2, 0.75  # BM25 defaults (the same as Lucene)
_FIELD_WEIGHTS = {"title": 2.0, "section": 1.5, "content": 1.0}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())  # like the standard analyzer: "AMM-32-11" splits


@dataclass
class _Stored:
    entry: IndexEntry
    acl: DocumentAcl
    document: Document


class InMemoryChunkIndex:
    def __init__(self, *, model_version: str, dimensions: int) -> None:
        self._model_version = model_version
        self._dims = dimensions
        self._rows: dict[tuple[str, str], _Stored] = {}  # (tenant_id, chunk_id) -> row
        self.fail_with: Exception | None = None  # set to simulate an outage on the next write
        self.write_calls = 0

    @property
    def model_version(self) -> str:
        return self._model_version

    async def ensure_ready(self) -> None:
        return None

    async def replace_document(self, document: Document, entries: Sequence[IndexEntry]) -> int:
        self.write_calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        seen: set[str] = set()
        for e in entries:
            c = e.chunk
            if (
                c.tenant_id != document.tenant_id
                or c.document_id != document.document_id
                or c.document_version != document.version
                or e.embedding.model_version != self._model_version
                or e.embedding.dimensions != self._dims
                or c.chunk_id in seen
            ):
                raise ContractError("invalid index entry")
            seen.add(c.chunk_id)
        newest = max(
            (
                r.entry.chunk.document_version
                for r in self._document_rows(document.tenant_id, document.document_id)
            ),
            default=None,
        )
        if newest is not None and newest > document.version:
            raise StaleIndexVersionError("a newer version of this document is already indexed")
        for key in [
            (r.entry.chunk.tenant_id, r.entry.chunk.chunk_id)
            for r in self._document_rows(document.tenant_id, document.document_id)
        ]:
            del self._rows[key]
        for e in entries:
            self._rows[(e.chunk.tenant_id, e.chunk.chunk_id)] = _Stored(e, document.acl, document)
        return len(entries)

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        for r in self._document_rows(tenant_id, document_id):
            del self._rows[(tenant_id, r.entry.chunk.chunk_id)]

    async def update_acl(self, tenant_id: str, document_id: str, acl: DocumentAcl) -> int:
        rows = self._document_rows(tenant_id, document_id)
        for r in rows:
            r.acl = acl
        return len(rows)

    async def count_chunks(self, tenant_id: str, document_id: str) -> int:
        return len(self._document_rows(tenant_id, document_id))

    # ---- search (ChunkSearcher port) --------------------------------------------------------

    async def lexical_search(
        self, query: str, *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        """BM25 per field, strongest field per term (like multi_match best_fields), summed."""
        _check_limit(limit)
        terms = set(_tokens(query))
        rows = self._visible(scope)
        if not terms or not rows:
            return []
        # Statistics come from the caller's visible rows only, so invisible documents cannot
        # influence (or be inferred from) a score.
        fields = [self._fields(r) for r in rows]
        n = len(rows)
        avg = {
            name: (sum(sum(f[name].values()) for f in fields) / n) or 1.0 for name in _FIELD_WEIGHTS
        }
        df = {
            (name, term): sum(1 for f in fields if term in f[name])
            for name in _FIELD_WEIGHTS
            for term in terms
        }

        def field_score(counts: Counter[str], name: str, term: str) -> float:
            tf = counts.get(term, 0)
            if not tf:
                return 0.0
            idf = math.log(1 + (n - df[(name, term)] + 0.5) / (df[(name, term)] + 0.5))
            length = sum(counts.values())
            return idf * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * length / avg[name]))

        scored: list[tuple[float, _Stored]] = []
        for r, f in zip(rows, fields, strict=True):
            total = sum(
                max(w * field_score(f[name], name, term) for name, w in _FIELD_WEIGHTS.items())
                for term in terms
            )
            if total > 0:
                scored.append((total, r))
        scored.sort(key=lambda t: (-t[0], t[1].entry.chunk.chunk_id))
        return [self._hit(r, score) for score, r in scored[:limit]]

    async def vector_search(
        self, vector: Sequence[float], *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        _check_limit(limit)
        if len(vector) != self._dims or not all(math.isfinite(x) for x in vector):
            raise ContractError("query vector has the wrong size or is not finite")
        norm_q = math.sqrt(sum(x * x for x in vector))
        if norm_q == 0:
            return []
        scored: list[tuple[float, _Stored]] = []
        for r in self._visible(scope):
            v = r.entry.embedding.vector
            norm_v = math.sqrt(sum(x * x for x in v)) or 1.0
            cosine = sum(a * b for a, b in zip(vector, v, strict=True)) / (norm_q * norm_v)
            scored.append(((1 + cosine) / 2, r))  # same 0..1 mapping OpenSearch uses for cosine
        scored.sort(key=lambda t: (-t[0], t[1].entry.chunk.chunk_id))
        return [self._hit(r, score) for score, r in scored[:limit]]

    def _visible(self, scope: SearchScope) -> list[_Stored]:
        if not scope.matches_any_acl:
            return []
        return [
            r
            for (t, _), r in self._rows.items()
            if t == scope.tenant_id and r.acl.permits(scope.user, scope.groups)
        ]

    @staticmethod
    def _fields(r: _Stored) -> dict[str, Counter[str]]:
        return {
            "title": Counter(_tokens(r.document.title)),
            "section": Counter(_tokens(" ".join(r.entry.chunk.section_path))),
            "content": Counter(_tokens(r.entry.chunk.content)),
        }

    @staticmethod
    def _hit(r: _Stored, score: float) -> RetrievedChunk:
        c = r.entry.chunk
        return RetrievedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            text=c.content,
            score=max(score, 0.0),
            document_version=c.document_version,
            title=r.document.title,
            source_uri=r.document.source_uri,
            section_path=tuple(c.section_path),
            page_number=c.page_number,
            content_hash=c.content_hash,
            kind=c.kind.value,
        )

    # ---- test helpers -----------------------------------------------------------------------

    def chunk_ids(self, tenant_id: str, document_id: str) -> list[str]:
        return sorted(r.entry.chunk.chunk_id for r in self._document_rows(tenant_id, document_id))

    def visible_chunk_ids(self, tenant_id: str, user: str, groups: frozenset[str]) -> list[str]:
        return sorted(
            cid
            for (t, cid), r in self._rows.items()
            if t == tenant_id and r.acl.permits(user, groups)
        )

    def _document_rows(self, tenant_id: str, document_id: str) -> list[_Stored]:
        return [
            r
            for (t, _), r in self._rows.items()
            if t == tenant_id and r.entry.chunk.document_id == document_id
        ]


def _check_limit(limit: int) -> None:
    if not 1 <= limit <= MAX_SEARCH_LIMIT:
        raise ContractError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}")
