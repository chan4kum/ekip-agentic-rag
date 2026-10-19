"""One behaviour spec for every ObjectStore adapter."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from ekip_contracts.documents import ContractError
from ekip_contracts.object_store import (
    ObjectIntegrityError,
    ObjectKey,
    ObjectKind,
    ObjectNotFoundError,
    ObjectStore,
    ObjectTooLargeError,
)
from ekip_persistence import InMemoryObjectStore, LocalFileObjectStore

RAW, IMG = ObjectKind.RAW_DOCUMENT, ObjectKind.IMAGE_ASSET
Factory = Callable[..., ObjectStore]


@pytest.fixture(params=["memory", "file"])
def make_store(request: pytest.FixtureRequest, tmp_path: Path) -> Factory:
    def build(**kwargs: int) -> ObjectStore:
        if request.param == "memory":
            return InMemoryObjectStore(**kwargs)
        return LocalFileObjectStore(tmp_path / "store", **kwargs)

    return build


async def test_put_get_round_trip_with_content_addressed_key(make_store: Factory) -> None:
    store = make_store()
    data = b"\x89PNG\r\n" + bytes(range(256))
    stored = await store.put("t1", IMG, data, content_type="image/png")
    assert stored.key.sha256 == hashlib.sha256(data).hexdigest()
    assert stored.size == len(data) and stored.content_type == "image/png"
    assert await store.get(stored.key) == data
    head = await store.head(stored.key)
    assert head is not None and head.size == len(data) and head.content_type == "image/png"


async def test_same_bytes_dedupe_within_a_tenant_first_content_type_wins(
    make_store: Factory,
) -> None:
    store = make_store()
    a = await store.put("t1", IMG, b"same", content_type="image/png")
    b = await store.put("t1", IMG, b"same", content_type="application/octet-stream")
    assert a.key == b.key and b.content_type == "image/png"


async def test_tenants_are_isolated_even_for_identical_bytes(make_store: Factory) -> None:
    store = make_store()
    a = await store.put("t1", RAW, b"shared bytes", content_type="text/plain")
    foreign = ObjectKey("t2", RAW, a.key.sha256)
    assert await store.head(foreign) is None  # no existence oracle across tenants
    with pytest.raises(ObjectNotFoundError):
        await store.get(foreign)
    b = await store.put("t2", RAW, b"shared bytes", content_type="text/plain")
    assert b.key != a.key
    assert await store.delete(b.key) is True
    assert await store.get(a.key) == b"shared bytes"  # deleting t2's copy leaves t1's alone


async def test_kinds_are_separate_namespaces(make_store: Factory) -> None:
    store = make_store()
    raw = await store.put("t1", RAW, b"x", content_type="text/plain")
    assert await store.head(ObjectKey("t1", IMG, raw.key.sha256)) is None


async def test_missing_object(make_store: Factory) -> None:
    store = make_store()
    key = ObjectKey("t1", RAW, hashlib.sha256(b"nope").hexdigest())
    assert await store.head(key) is None
    assert await store.delete(key) is False
    with pytest.raises(ObjectNotFoundError):
        await store.get(key)


async def test_delete_removes_the_object(make_store: Factory) -> None:
    store = make_store()
    stored = await store.put("t1", RAW, b"bye", content_type="text/plain")
    assert await store.delete(stored.key) is True
    assert await store.head(stored.key) is None
    assert await store.delete(stored.key) is False


async def test_empty_object_is_allowed(make_store: Factory) -> None:
    store = make_store()
    stored = await store.put("t1", RAW, b"", content_type="text/plain")
    assert await store.get(stored.key) == b"" and stored.size == 0


async def test_size_limit(make_store: Factory) -> None:
    store = make_store(max_bytes=10)
    await store.put("t1", RAW, b"x" * 10, content_type="text/plain")
    with pytest.raises(ObjectTooLargeError):
        await store.put("t1", RAW, b"x" * 11, content_type="text/plain")


@pytest.mark.parametrize("tenant", ["../evil", "a/b", "", "t1/../t2"])
async def test_hostile_tenant_ids_are_rejected_before_touching_storage(
    make_store: Factory, tenant: str
) -> None:
    with pytest.raises(ContractError):
        await make_store().put(tenant, RAW, b"x", content_type="text/plain")


@pytest.mark.parametrize("content_type", ["", "png", "image/png; x=y", "a b/c", "text/plain\n"])
async def test_bad_content_types_are_rejected(make_store: Factory, content_type: str) -> None:
    with pytest.raises(ContractError):
        await make_store().put("t1", RAW, b"x", content_type=content_type)


async def test_concurrent_identical_puts_are_safe(make_store: Factory) -> None:
    import asyncio

    store = make_store()
    results = await asyncio.gather(
        *(store.put("t1", RAW, b"racing", content_type="text/plain") for _ in range(20))
    )
    assert len({r.key for r in results}) == 1
    assert await store.get(results[0].key) == b"racing"


async def test_file_store_detects_corruption(tmp_path: Path) -> None:
    store = LocalFileObjectStore(tmp_path / "s")
    stored = await store.put("t1", RAW, b"original", content_type="text/plain")
    (tmp_path / "s" / stored.key.path).write_bytes(b"tampered")
    with pytest.raises(ObjectIntegrityError):
        await store.get(stored.key)


async def test_file_store_layout_is_tenant_prefixed_and_leaves_no_temp_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "s"
    store = LocalFileObjectStore(root)
    stored = await store.put("t1", IMG, b"img", content_type="image/png")
    assert (root / stored.key.path).is_file()
    assert all(
        p.name.startswith(("tenants", "t1", "assets")) or len(p.name) in (2, 64, 69)
        for p in root.rglob("*")
        if p.is_dir() or p.is_file()
    )
    assert not list(root.rglob(".tmp-*"))
