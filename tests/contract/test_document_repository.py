"""One behaviour spec, run against every DocumentRepository implementation.

The in-memory twin always runs. The Postgres adapter runs with RUN_INTEGRATION=1 against the
Compose stack (migrations are applied first). A behaviour difference between the two is a bug.
"""

import asyncio
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Protocol

import pytest
from ekip_contracts.assets import ImageAsset
from ekip_contracts.document_store import (
    AssetRepository,
    DocumentNotFoundError,
    DocumentRepository,
    ImmutableVersionError,
    InvalidTransitionError,
    VersionConflictError,
)
from ekip_contracts.documents import (
    Chunk,
    ChunkKind,
    ContractError,
    Document,
    DocumentAcl,
    DocumentStatus,
    sha256_hex,
)
from ekip_persistence import InMemoryDocumentRepository, PostgresDocumentRepository, create_pool
from tests_paths import ROOT


class Repo(DocumentRepository, AssetRepository, Protocol):
    """Both ports, as the real adapters provide them."""


LIVE = os.environ.get("RUN_INTEGRATION") == "1"
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://ekip:ekip@localhost:5432/ekip")
S = DocumentStatus
NOW = datetime(2026, 10, 19, 6, 30, tzinfo=UTC)


@pytest.fixture(scope="session")
def migrated() -> None:
    if not LIVE:
        return
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(ROOT / "packages/persistence/alembic.ini"),
            "upgrade",
            "head",
        ],
        env={**os.environ, "DATABASE_URL": DATABASE_URL},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.integration)])
async def repo(request: pytest.FixtureRequest, migrated: None) -> AsyncIterator[Repo]:
    if request.param == "memory":
        yield InMemoryDocumentRepository()
        return
    if not LIVE:
        pytest.skip("needs live Compose stack (RUN_INTEGRATION=1)")
    pool = await create_pool(DATABASE_URL, max_size=10)
    try:
        yield PostgresDocumentRepository(pool)
        async with pool.acquire() as conn:  # remove everything the tests created
            await conn.execute("DELETE FROM documents WHERE tenant_id LIKE 'rt-%'")
    finally:
        await pool.close()


@pytest.fixture
def tenant() -> Callable[[], str]:
    return lambda: f"rt-{uuid.uuid4().hex[:12]}"


def doc(
    tenant_id: str,
    *,
    version: int = 1,
    content: bytes = b"v1",
    sid: str = "a/b.pdf",
    source_version: str = "r1",
) -> Document:
    return Document.create(
        tenant_id=tenant_id,
        source="sharepoint",
        source_document_id=sid,
        version=version,
        source_version=source_version,
        content=content,
        title="Manual",
        mime_type="application/pdf",
        source_uri="https://example.test/m.pdf",
        acl=DocumentAcl("acl-1", frozenset({"alice"}), frozenset({"eng", "ops"})),
        last_modified=NOW,
    )


def chunks_for(d: Document, n: int = 3) -> list[Chunk]:
    return [
        Chunk.create(
            document=d,
            ordinal=i,
            content=f"chunk {i} ünïcode ✓",
            section_path=("Ch 1", f"S{i}"),
            page_number=i + 1,
        )
        for i in range(n)
    ]


async def advance(repo: Repo, d: Document, *path: DocumentStatus) -> None:
    for status in path:
        await repo.transition(d.tenant_id, d.document_id, d.version, status)


HAPPY = (S.PARSING, S.CHUNKING, S.EMBEDDING, S.INDEXING, S.INDEXED)


# ---- register / get ---------------------------------------------------------------------------
async def test_register_then_get_round_trips_every_field(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    stored = await repo.register(d)
    assert stored == d
    assert await repo.get(d.tenant_id, d.document_id, 1) == d  # incl. ACL sets and tz-aware time


async def test_register_is_idempotent_for_identical_input(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    await advance(repo, d, S.PARSING)
    again = await repo.register(d)
    assert again.status is S.PARSING  # a retry never resets progress


async def test_register_conflicting_content_for_same_version_is_rejected(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    t = tenant()
    await repo.register(doc(t))
    with pytest.raises(VersionConflictError):
        await repo.register(doc(t, content=b"different"))
    with pytest.raises(VersionConflictError):
        await repo.register(doc(t, source_version="r2"))


async def test_only_pending_documents_can_be_registered(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    from dataclasses import replace

    with pytest.raises(ContractError):
        await repo.register(replace(doc(tenant()), status=S.INDEXED))


async def test_concurrent_registration_has_one_row_and_no_errors(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    results = await asyncio.gather(*(repo.register(d) for _ in range(20)))
    assert all(r.content_hash == d.content_hash for r in results)


async def test_latest_version(repo: Repo, tenant: Callable[[], str]) -> None:
    t = tenant()
    d1 = doc(t)
    assert await repo.latest_version(t, d1.document_id) is None
    await repo.register(d1)
    await repo.register(doc(t, version=3, content=b"v3"))
    await repo.register(doc(t, version=2, content=b"v2"))
    assert await repo.latest_version(t, d1.document_id) == 3


# ---- tenant isolation -------------------------------------------------------------------------
async def test_another_tenant_cannot_see_or_touch_a_document(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    a, b = tenant(), tenant()
    da = doc(a)
    await repo.register(da)
    assert await repo.get(b, da.document_id, 1) is None
    assert await repo.latest_version(b, da.document_id) is None
    with pytest.raises(DocumentNotFoundError):
        await repo.transition(b, da.document_id, 1, S.PARSING)
    with pytest.raises(DocumentNotFoundError):
        await repo.replace_chunks(b, da.document_id, 1, [])
    assert await repo.get_chunks(b, da.document_id, 1) == []
    assert (await repo.get(a, da.document_id, 1)) is not None  # untouched


async def test_same_source_document_in_two_tenants_are_independent(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    a, b = tenant(), tenant()
    da, db = doc(a), doc(b)
    assert da.document_id != db.document_id
    await repo.register(da)
    await repo.register(db)
    await advance(repo, da, S.PARSING)
    stored_b = await repo.get(b, db.document_id, 1)
    assert stored_b is not None and stored_b.status is S.PENDING


# ---- status transitions -----------------------------------------------------------------------
async def test_happy_path_reaches_indexed_and_becomes_searchable(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    await advance(repo, d, *HAPPY)
    final = await repo.get(d.tenant_id, d.document_id, 1)
    assert final is not None and final.is_searchable


@pytest.mark.parametrize(
    ("path", "illegal"),
    [
        ((), S.INDEXED),  # skipping every stage
        ((S.PARSING,), S.EMBEDDING),  # skipping chunking
        ((S.PARSING,), S.PARSING),  # no self-transition
        (HAPPY, S.PARSING),  # an indexed version never goes back
        (HAPPY, S.FAILED),
        ((S.PARSING, S.FAILED), S.INDEXED),  # failed must restart, not teleport
    ],
)
async def test_illegal_transitions_are_rejected_and_change_nothing(
    repo: Repo, tenant: Callable[[], str], path: tuple[S, ...], illegal: S
) -> None:
    d = doc(tenant())
    await repo.register(d)
    await advance(repo, d, *path)
    before = await repo.get(d.tenant_id, d.document_id, 1)
    with pytest.raises(InvalidTransitionError):
        await repo.transition(d.tenant_id, d.document_id, 1, illegal)
    assert await repo.get(d.tenant_id, d.document_id, 1) == before


async def test_failed_records_a_reason_and_can_resume_at_a_later_stage(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    await advance(repo, d, S.PARSING, S.CHUNKING, S.EMBEDDING, S.INDEXING)
    failed = await repo.transition(
        d.tenant_id, d.document_id, 1, S.FAILED, failure_reason="index timeout"
    )
    assert failed.status is S.FAILED and failed.failure_reason == "index timeout"
    resumed = await repo.transition(d.tenant_id, d.document_id, 1, S.INDEXING)
    assert resumed.status is S.INDEXING and resumed.failure_reason is None  # reason cleared


async def test_failure_reason_is_ignored_for_non_failed_targets(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    moved = await repo.transition(d.tenant_id, d.document_id, 1, S.PARSING, failure_reason="x")
    assert moved.failure_reason is None


async def test_deleted_is_terminal(repo: Repo, tenant: Callable[[], str]) -> None:
    d = doc(tenant())
    await repo.register(d)
    await advance(repo, d, *HAPPY, S.DELETED)
    for target in S:
        with pytest.raises(InvalidTransitionError):
            await repo.transition(d.tenant_id, d.document_id, 1, target)


async def test_transition_of_a_missing_version_is_not_found(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    with pytest.raises(DocumentNotFoundError):
        await repo.transition(d.tenant_id, d.document_id, 1, S.PARSING)


async def test_two_workers_racing_for_the_same_step_exactly_one_wins(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    results = await asyncio.gather(
        *(repo.transition(d.tenant_id, d.document_id, 1, S.PARSING) for _ in range(10)),
        return_exceptions=True,
    )
    wins = [r for r in results if isinstance(r, Document)]
    losses = [r for r in results if isinstance(r, InvalidTransitionError)]
    assert len(wins) == 1 and len(losses) == 9


# ---- chunks -----------------------------------------------------------------------------------
async def test_chunks_round_trip_in_ordinal_order(repo: Repo, tenant: Callable[[], str]) -> None:
    d = doc(tenant())
    await repo.register(d)
    cs = chunks_for(d, 5)
    await repo.replace_chunks(d.tenant_id, d.document_id, 1, list(reversed(cs)))
    assert await repo.get_chunks(d.tenant_id, d.document_id, 1) == cs


async def test_replace_chunks_replaces_the_whole_set(repo: Repo, tenant: Callable[[], str]) -> None:
    d = doc(tenant())
    await repo.register(d)
    await repo.replace_chunks(d.tenant_id, d.document_id, 1, chunks_for(d, 5))
    await repo.replace_chunks(d.tenant_id, d.document_id, 1, chunks_for(d, 2))
    assert len(await repo.get_chunks(d.tenant_id, d.document_id, 1)) == 2
    await repo.replace_chunks(d.tenant_id, d.document_id, 1, [])
    assert await repo.get_chunks(d.tenant_id, d.document_id, 1) == []


async def test_versions_keep_separate_chunk_sets(repo: Repo, tenant: Callable[[], str]) -> None:
    t = tenant()
    d1, d2 = doc(t), doc(t, version=2, content=b"v2")
    await repo.register(d1)
    await repo.register(d2)
    await repo.replace_chunks(t, d1.document_id, 1, chunks_for(d1, 3))
    await repo.replace_chunks(t, d2.document_id, 2, chunks_for(d2, 2))
    assert len(await repo.get_chunks(t, d1.document_id, 1)) == 3
    assert len(await repo.get_chunks(t, d2.document_id, 2)) == 2


async def test_chunks_of_another_version_or_tenant_are_rejected_atomically(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    a, b = tenant(), tenant()
    da, db = doc(a), doc(b)
    await repo.register(da)
    await repo.register(db)
    good = chunks_for(da, 2)
    await repo.replace_chunks(a, da.document_id, 1, good)
    with pytest.raises(ContractError):  # a chunk belonging to tenant b smuggled into a's set
        await repo.replace_chunks(a, da.document_id, 1, [*chunks_for(da, 1), *chunks_for(db, 1)])
    with pytest.raises(ContractError):  # duplicate ordinals
        c = chunks_for(da, 1)[0]
        await repo.replace_chunks(a, da.document_id, 1, [c, c])
    assert await repo.get_chunks(a, da.document_id, 1) == good  # failed replace changed nothing


async def test_chunks_are_frozen_once_indexed_or_deleted(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    await repo.replace_chunks(d.tenant_id, d.document_id, 1, chunks_for(d, 2))
    await advance(repo, d, *HAPPY)
    with pytest.raises(ImmutableVersionError):
        await repo.replace_chunks(d.tenant_id, d.document_id, 1, chunks_for(d, 1))
    assert len(await repo.get_chunks(d.tenant_id, d.document_id, 1)) == 2


async def test_chunks_for_unknown_version_are_not_found(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    with pytest.raises(DocumentNotFoundError):
        await repo.replace_chunks(d.tenant_id, d.document_id, 1, [])


async def test_concurrent_replacers_leave_one_complete_set(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    sets = [chunks_for(d, n) for n in (2, 3, 4, 5)]
    await asyncio.gather(*(repo.replace_chunks(d.tenant_id, d.document_id, 1, s) for s in sets))
    final = await repo.get_chunks(d.tenant_id, d.document_id, 1)
    assert final in sets  # exactly one writer's complete set, never a mixture


# ---- image chunks and assets -------------------------------------------------------------------
def asset_for(d: Document, ordinal: int = 0, **overrides: object) -> ImageAsset:
    args: dict[str, object] = {
        "tenant_id": d.tenant_id,
        "document_id": d.document_id,
        "document_version": d.version,
        "ordinal": ordinal,
        "sha256": sha256_hex(f"img{ordinal}"),
        "mime_type": "image/png",
        "byte_size": 100 + ordinal,
        "width": 640,
        "height": 480,
        "page_number": 2,
        "bbox": (0.1, 0.2, 0.9, 0.8),
        "nearby_chunk_ordinal": 1,
        "caption": "Figure 1",
        "alt_text": "torque map",
        "ocr_text": "40 Nm",
        "description": "bolt order diagram",
        "description_model_version": "vision-x@v1",
    }
    args.update(overrides)
    return ImageAsset(**args)  # type: ignore[arg-type]


async def test_image_chunks_round_trip_with_their_asset_reference(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    a = asset_for(d)
    image_chunk = Chunk.create(
        document=d, ordinal=0, content=a.search_text, kind=ChunkKind.IMAGE, asset_sha256=a.sha256
    )
    text_chunk = Chunk.create(document=d, ordinal=1, content="plain text")
    await repo.replace_chunks(d.tenant_id, d.document_id, 1, [image_chunk, text_chunk])
    got = await repo.get_chunks(d.tenant_id, d.document_id, 1)
    assert got == [image_chunk, text_chunk]
    assert got[0].kind is ChunkKind.IMAGE and got[1].asset_sha256 is None


async def test_assets_round_trip_every_field_in_ordinal_order(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    d = doc(tenant())
    await repo.register(d)
    assets = [asset_for(d, 0), asset_for(d, 1, bbox=None, page_number=None, caption=None)]
    await repo.replace_assets(d.tenant_id, d.document_id, 1, list(reversed(assets)))
    assert await repo.list_assets(d.tenant_id, d.document_id, 1) == assets
    assert await repo.get_asset(d.tenant_id, d.document_id, 1, assets[1].sha256) == assets[1]
    assert await repo.get_asset(d.tenant_id, d.document_id, 1, sha256_hex("other")) is None


async def test_replace_assets_replaces_and_is_atomic_on_bad_input(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    a, b = tenant(), tenant()
    da, db = doc(a), doc(b)
    await repo.register(da)
    await repo.register(db)
    first = [asset_for(da, 0), asset_for(da, 1)]
    await repo.replace_assets(a, da.document_id, 1, first)
    with pytest.raises(ContractError):  # another tenant's asset smuggled in
        await repo.replace_assets(a, da.document_id, 1, [asset_for(da, 0), asset_for(db, 1)])
    with pytest.raises(ContractError):  # duplicate ordinal
        await repo.replace_assets(a, da.document_id, 1, [asset_for(da, 0), asset_for(da, 0)])
    assert await repo.list_assets(a, da.document_id, 1) == first
    await repo.replace_assets(a, da.document_id, 1, [asset_for(da, 5)])
    assert [x.ordinal for x in await repo.list_assets(a, da.document_id, 1)] == [5]


async def test_assets_are_tenant_isolated_and_frozen_once_indexed(
    repo: Repo, tenant: Callable[[], str]
) -> None:
    a, b = tenant(), tenant()
    d = doc(a)
    await repo.register(d)
    await repo.replace_assets(a, d.document_id, 1, [asset_for(d)])
    assert await repo.list_assets(b, d.document_id, 1) == []
    assert await repo.get_asset(b, d.document_id, 1, asset_for(d).sha256) is None
    with pytest.raises(DocumentNotFoundError):
        await repo.replace_assets(b, d.document_id, 1, [])
    await advance(repo, d, *HAPPY)
    with pytest.raises(ImmutableVersionError):
        await repo.replace_assets(a, d.document_id, 1, [])
    assert len(await repo.list_assets(a, d.document_id, 1)) == 1
