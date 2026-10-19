"""A small seeded corpus for local runs, the smoke test and API tests (backend = "memory").

It goes through the SAME index/ACL rules as production: every document has an ACL, so the demo also
shows that two users in one tenant can see different results. Nothing here is real data.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ekip_contracts.documents import Chunk, ChunkEmbedding, Document, DocumentAcl
from ekip_contracts.embedding import Embedder, EmbeddingTask
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_contracts.search_index import IndexEntry
from ekip_llm_gateway.hashing import HashingEmbedder
from ekip_search.memory import InMemoryChunkIndex

DEMO_DIMS = 64


@dataclass(frozen=True)
class DemoDocument:
    tenant: str
    source_id: str
    title: str
    groups: tuple[str, ...]
    passages: tuple[tuple[str, str], ...]  # (section, text)
    users: tuple[str, ...] = ()


DEMO_DOCUMENTS: tuple[DemoDocument, ...] = (
    DemoDocument(
        "tenant-dev", "boeing-mx-v1", "Maintenance manual v1", ("eng", "ops"),
        (("Landing gear", "Fastener torque must be verified during the scheduled inspection "
                          "of the landing gear."),),
    ),
    DemoDocument(
        "tenant-dev", "boeing-mx-v2", "Maintenance manual v2", ("eng", "ops"),
        (
            ("Landing gear", "Revised procedure: torque verification now requires a calibrated "
                             "wrench and a second inspector sign-off."),
            ("Hydraulics", "Hydraulic line inspection interval reduced from 600 to 400 "
                          "flight hours."),
        ),
    ),
    DemoDocument(
        "tenant-dev", "cabin-handbook", "Cabin handbook", ("eng", "ops"),
        (("Galley", "Cabin lighting replacement procedure for the forward galley."),),
    ),
    DemoDocument(  # only the "ops" group: engineers in the same tenant must NOT see this
        "tenant-dev", "ops-audit", "Operations audit findings", ("ops",),
        (("Findings", "Confidential audit finding: torque inspection records were incomplete."),),
    ),
    DemoDocument(
        "tenant-other", "other-doc", "Other tenant schedule", ("eng",),
        (
            ("Schedule", "Confidential torque inspection schedule belonging to another tenant."),
            ("Payroll", "Payroll processing calendar."),
        ),
    ),
)  # fmt: skip


async def seed_demo_index(index: InMemoryChunkIndex, embedder: Embedder) -> None:
    for spec in DEMO_DOCUMENTS:
        doc = Document.create(
            tenant_id=spec.tenant,
            source="demo",
            source_document_id=spec.source_id,
            version=1,
            source_version="v1",
            content=spec.source_id.encode(),
            title=spec.title,
            mime_type="text/plain",
            source_uri=f"https://demo.test/{spec.tenant}/{spec.source_id}",
            acl=DocumentAcl("demo-acl", frozenset(spec.users), frozenset(spec.groups)),
            last_modified=datetime(2026, 10, 19, tzinfo=UTC),
        )
        chunks = [
            Chunk.create(document=doc, ordinal=i, content=text, section_path=(section,))
            for i, (section, text) in enumerate(spec.passages)
        ]
        vectors = await embedder.embed([c.content for c in chunks], task=EmbeddingTask.DOCUMENT)
        entries: Sequence[IndexEntry] = [
            IndexEntry(c, ChunkEmbedding(c.chunk_id, c.content_hash, embedder.model_version, v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        await index.replace_document(doc, entries)


class SeededDemoSearcher:
    """ChunkSearcher over the demo corpus. Seeds on first use, because ASGI test clients do not
    run the app lifespan and a sync constructor cannot await."""

    def __init__(self) -> None:
        self.embedder = HashingEmbedder(DEMO_DIMS)
        self._index = InMemoryChunkIndex(
            model_version=self.embedder.model_version, dimensions=DEMO_DIMS
        )
        self._lock = asyncio.Lock()
        self._seeded = False

    async def _ready(self) -> InMemoryChunkIndex:
        if not self._seeded:
            async with self._lock:
                if not self._seeded:
                    await seed_demo_index(self._index, self.embedder)
                    self._seeded = True
        return self._index

    async def lexical_search(
        self, query: str, *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        return await (await self._ready()).lexical_search(query, scope=scope, limit=limit)

    async def vector_search(
        self, vector: Sequence[float], *, scope: SearchScope, limit: int
    ) -> list[RetrievedChunk]:
        return await (await self._ready()).vector_search(vector, scope=scope, limit=limit)
