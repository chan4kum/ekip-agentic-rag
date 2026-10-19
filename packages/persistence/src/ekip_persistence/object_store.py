"""ObjectStore adapters: in-memory (tests) and local filesystem (dev / single node).

Both compute the key from the bytes (sha256), enforce a size limit, and verify integrity on read.
An S3-compatible adapter arrives with the cloud deployment (same port, same tests).
"""

import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path

from ekip_contracts.object_store import (
    ObjectIntegrityError,
    ObjectKey,
    ObjectKind,
    ObjectNotFoundError,
    ObjectTooLargeError,
    StoredObject,
    check_content_type,
)

DEFAULT_MAX_BYTES = 100 * 1024 * 1024


def _key_for(tenant_id: str, kind: ObjectKind, data: bytes) -> ObjectKey:
    return ObjectKey(tenant_id, kind, hashlib.sha256(data).hexdigest())


class InMemoryObjectStore:
    def __init__(self, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._max = max_bytes
        self._objects: dict[ObjectKey, tuple[bytes, str]] = {}

    async def put(
        self, tenant_id: str, kind: ObjectKind, data: bytes, *, content_type: str
    ) -> StoredObject:
        check_content_type(content_type)
        if len(data) > self._max:
            raise ObjectTooLargeError("object exceeds the size limit")
        key = _key_for(tenant_id, kind, data)
        _, stored_type = self._objects.setdefault(key, (data, content_type))  # first writer wins
        return StoredObject(key, len(data), stored_type)

    async def get(self, key: ObjectKey) -> bytes:
        found = self._objects.get(key)
        if found is None:
            raise ObjectNotFoundError("object not found")
        return found[0]

    async def head(self, key: ObjectKey) -> StoredObject | None:
        found = self._objects.get(key)
        return None if found is None else StoredObject(key, len(found[0]), found[1])

    async def delete(self, key: ObjectKey) -> bool:
        return self._objects.pop(key, None) is not None


class LocalFileObjectStore:
    """Files under `root`, laid out exactly like the key path. Writes are atomic (temp + rename)."""

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._root = root.resolve()
        self._max = max_bytes
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: ObjectKey) -> Path:
        path = (self._root / key.path).resolve()
        if not path.is_relative_to(self._root):  # unreachable with validated keys; belt and braces
            raise ObjectNotFoundError("object not found")
        return path

    async def put(
        self, tenant_id: str, kind: ObjectKind, data: bytes, *, content_type: str
    ) -> StoredObject:
        check_content_type(content_type)
        if len(data) > self._max:
            raise ObjectTooLargeError("object exceeds the size limit")
        key = _key_for(tenant_id, kind, data)
        return await asyncio.to_thread(self._put_sync, key, data, content_type)

    def _put_sync(self, key: ObjectKey, data: bytes, content_type: str) -> StoredObject:
        path = self._path(key)
        meta = path.with_suffix(".meta")
        if path.exists() and meta.exists():
            stored = json.loads(meta.read_text())
            return StoredObject(key, len(data), stored["content_type"])  # already there: dedupe
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        self._atomic_write(path, data)
        self._atomic_write(meta, json.dumps({"content_type": content_type}).encode())
        return StoredObject(key, len(data), content_type)

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        tmp = target.with_name(f".tmp-{uuid.uuid4().hex}")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)

    async def get(self, key: ObjectKey) -> bytes:
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: ObjectKey) -> bytes:
        try:
            data = self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise ObjectNotFoundError("object not found") from exc
        if hashlib.sha256(data).hexdigest() != key.sha256:
            raise ObjectIntegrityError("stored bytes do not match their key")
        return data

    async def head(self, key: ObjectKey) -> StoredObject | None:
        return await asyncio.to_thread(self._head_sync, key)

    def _head_sync(self, key: ObjectKey) -> StoredObject | None:
        path = self._path(key)
        meta = path.with_suffix(".meta")
        if not (path.exists() and meta.exists()):
            return None
        content_type = json.loads(meta.read_text())["content_type"]
        return StoredObject(key, path.stat().st_size, content_type)

    async def delete(self, key: ObjectKey) -> bool:
        return await asyncio.to_thread(self._delete_sync, key)

    def _delete_sync(self, key: ObjectKey) -> bool:
        path = self._path(key)
        existed = path.exists()
        path.unlink(missing_ok=True)
        path.with_suffix(".meta").unlink(missing_ok=True)
        return existed
