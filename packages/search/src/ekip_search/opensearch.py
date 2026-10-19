"""OpenSearch implementation of the ChunkIndex port (httpx, no SDK).

Failure behaviour: transient errors (429/502/503/504, network) are retried with jittered backoff;
everything else raises SearchIndexError. Error messages carry the HTTP status only - never the
response body, which can echo document text.
"""

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from urllib.parse import quote

import httpx
from ekip_contracts.documents import ContractError, Document, DocumentAcl
from ekip_contracts.search_index import (
    IndexEntry,
    SearchIndexError,
    SearchIndexUnavailableError,
    StaleIndexVersionError,
)

from ekip_search.mapping import MAPPING_VERSION, index_body, index_name_for

_RETRYABLE = {429, 502, 503, 504}
BULK_BATCH = 200


class OpenSearchChunkIndex:
    def __init__(
        self,
        base_url: str,
        *,
        model_version: str,
        dimensions: int,
        auth: tuple[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        shards: int = 1,
        replicas: int = 0,
        index_name: str | None = None,
        max_attempts: int = 4,
        timeout_s: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = lambda: random.uniform(0.5, 1.5),  # noqa: S311 - not security
    ) -> None:
        self._model_version = model_version
        self._dims = dimensions
        self._index = index_name or index_name_for(model_version)
        self._shards, self._replicas = shards, replicas
        self._base = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_s, auth=auth, trust_env=True)
        self._owns_client = client is None
        self._attempts = max_attempts
        self._sleep = sleep
        self._jitter = jitter

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def index_name(self) -> str:
        return self._index

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---- port -------------------------------------------------------------------------------

    async def ensure_ready(self) -> None:
        r = await self._call("GET", f"/{self._index}/_mapping", ok={200, 404})
        if r.status_code == 404:
            body = index_body(
                dimensions=self._dims,
                model_version=self._model_version,
                shards=self._shards,
                replicas=self._replicas,
            )
            created = await self._call("PUT", f"/{self._index}", json_body=body, ok={200, 400})
            if created.status_code == 200:
                return
            if "resource_already_exists" not in created.text:  # lost a creation race: fine
                raise SearchIndexError("index creation rejected")
            r = await self._call("GET", f"/{self._index}/_mapping", ok={200})
        meta = (r.json().get(self._index, {}).get("mappings", {}).get("_meta")) or {}
        if (
            meta.get("mapping_version") != MAPPING_VERSION
            or meta.get("embedding_model_version") != self._model_version
            or meta.get("dimensions") != self._dims
        ):
            raise SearchIndexError("existing index has an incompatible mapping; reindex required")

    async def replace_document(self, document: Document, entries: Sequence[IndexEntry]) -> int:
        self._validate(document, entries)
        newest = await self._newest_version(document.tenant_id, document.document_id)
        if newest is not None and newest > document.version:
            raise StaleIndexVersionError("a newer version of this document is already indexed")
        for start in range(0, len(entries), BULK_BATCH):
            await self._bulk(document, entries[start : start + BULK_BATCH])
        await self._call("POST", f"/{self._index}/_refresh", ok={200})
        keep = [e.chunk.chunk_id for e in entries]
        stale = {
            "bool": {
                "filter": [
                    {"term": {"tenant_id": document.tenant_id}},
                    {"term": {"document_id": document.document_id}},
                ],
                "must": {
                    "bool": {
                        "should": [
                            {"range": {"document_version": {"lt": document.version}}},
                            {
                                "bool": {
                                    "filter": [{"term": {"document_version": document.version}}],
                                    "must_not": [{"ids": {"values": keep}}],
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
        }
        await self._delete_by_query(document.tenant_id, stale)
        return await self.count_chunks(document.tenant_id, document.document_id)

    async def delete_document(self, tenant_id: str, document_id: str) -> None:
        await self._delete_by_query(tenant_id, self._document_filter(tenant_id, document_id))

    async def update_acl(self, tenant_id: str, document_id: str, acl: DocumentAcl) -> int:
        body = {
            "query": self._document_filter(tenant_id, document_id),
            "script": {
                "lang": "painless",
                "source": (
                    "ctx._source.allowed_users = params.users;"
                    "ctx._source.allowed_groups = params.groups;"
                    "ctx._source.acl_version = params.version;"
                ),
                "params": {
                    "users": sorted(acl.allowed_users),
                    "groups": sorted(acl.allowed_groups),
                    "version": acl.acl_version,
                },
            },
        }
        r = await self._call(
            "POST",
            f"/{self._index}/_update_by_query",
            params={"routing": tenant_id, "refresh": "true", "conflicts": "proceed"},
            json_body=body,
            ok={200},
        )
        return int(r.json().get("updated", 0))

    async def count_chunks(self, tenant_id: str, document_id: str) -> int:
        r = await self._call(
            "POST",
            f"/{self._index}/_count",
            params={"routing": tenant_id},
            json_body={"query": self._document_filter(tenant_id, document_id)},
            ok={200},
        )
        return int(r.json()["count"])

    # ---- internals --------------------------------------------------------------------------

    @staticmethod
    def _document_filter(tenant_id: str, document_id: str) -> dict[str, Any]:
        return {
            "bool": {
                "filter": [
                    {"term": {"tenant_id": tenant_id}},
                    {"term": {"document_id": document_id}},
                ]
            }
        }

    def _validate(self, document: Document, entries: Sequence[IndexEntry]) -> None:
        seen: set[str] = set()
        for e in entries:
            c = e.chunk
            if (
                c.tenant_id != document.tenant_id
                or c.document_id != document.document_id
                or c.document_version != document.version
            ):
                raise ContractError("chunk does not belong to this document version")
            if e.embedding.model_version != self._model_version:
                raise ContractError("embedding was produced by a different model version")
            if e.embedding.dimensions != self._dims:
                raise ContractError("embedding has the wrong number of dimensions")
            if c.chunk_id in seen:
                raise ContractError("duplicate chunk in one publication")
            seen.add(c.chunk_id)

    async def _newest_version(self, tenant_id: str, document_id: str) -> int | None:
        r = await self._call(
            "POST",
            f"/{self._index}/_search",
            params={"routing": tenant_id},
            json_body={
                "size": 0,
                "query": self._document_filter(tenant_id, document_id),
                "aggs": {"newest": {"max": {"field": "document_version"}}},
            },
            ok={200},
        )
        value = r.json()["aggregations"]["newest"]["value"]
        return None if value is None else int(value)

    async def _bulk(self, document: Document, batch: Sequence[IndexEntry]) -> None:
        lines: list[str] = []
        for e in batch:
            c = e.chunk
            lines.append(
                json.dumps(
                    {"index": {"_index": self._index, "_id": c.chunk_id, "routing": c.tenant_id}}
                )
            )
            lines.append(json.dumps(self._source(document, e), allow_nan=False))
        r = await self._call(
            "POST",
            "/_bulk",
            params={"refresh": "false"},
            content="\n".join(lines) + "\n",
            ok={200},
        )
        if r.json().get("errors"):
            raise SearchIndexError("index rejected one or more chunks")

    @staticmethod
    def _source(document: Document, e: IndexEntry) -> dict[str, Any]:
        c = e.chunk
        return {
            "tenant_id": c.tenant_id,
            "document_id": c.document_id,
            "chunk_id": c.chunk_id,
            "document_version": c.document_version,
            "ordinal": c.ordinal,
            "kind": c.kind.value,
            "source": document.source,
            "mime_type": document.mime_type,
            "source_uri": document.source_uri,
            "content_hash": c.content_hash,
            "asset_sha256": c.asset_sha256,
            "page_number": c.page_number,
            "last_modified": document.last_modified.isoformat(),
            "allowed_users": sorted(document.acl.allowed_users),
            "allowed_groups": sorted(document.acl.allowed_groups),
            "acl_version": document.acl.acl_version,
            "title": document.title,
            "section_path": " > ".join(c.section_path),
            "content": c.content,
            "embedding_model_version": e.embedding.model_version,
            "embedding": list(e.embedding.vector),
        }

    async def _delete_by_query(self, tenant_id: str, query: dict[str, Any]) -> None:
        await self._call(
            "POST",
            f"/{self._index}/_delete_by_query",
            params={"routing": tenant_id, "refresh": "true", "conflicts": "proceed"},
            json_body={"query": query},
            ok={200},
        )

    async def _call(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        content: str | None = None,
        ok: set[int],
    ) -> httpx.Response:
        url = self._base + quote(path, safe="/_,*-")
        headers = {"content-type": "application/x-ndjson"} if content is not None else None
        last = "unknown"
        for attempt in range(self._attempts):
            try:
                r = await self._client.request(
                    method, url, params=params, json=json_body, content=content, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError):
                last = "network error"
            else:
                if r.status_code in ok:
                    return r
                if r.status_code not in _RETRYABLE:
                    raise SearchIndexError(
                        f"search index rejected the request (HTTP {r.status_code})"
                    )
                last = f"HTTP {r.status_code}"
            if attempt + 1 < self._attempts:
                await self._sleep(min(0.5 * 2**attempt * self._jitter(), 10.0))
        raise SearchIndexUnavailableError(f"search index unavailable ({last})")
