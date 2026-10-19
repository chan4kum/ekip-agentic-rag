from dataclasses import replace

import pytest
from ekip_contracts.assets import ImageAsset
from ekip_contracts.documents import (
    Chunk,
    ChunkKind,
    ContractError,
    chunk_id_for,
    document_id_for,
    sha256_hex,
)
from ekip_contracts.object_store import ObjectKey, ObjectKind, StoredObject

DOC = document_id_for("t1", "sharepoint", "x")
SHA = sha256_hex(b"png-bytes")


def make_asset(**overrides: object) -> ImageAsset:
    args: dict[str, object] = {
        "tenant_id": "t1",
        "document_id": DOC,
        "document_version": 1,
        "ordinal": 0,
        "sha256": SHA,
        "mime_type": "image/png",
        "byte_size": 1234,
        "width": 640,
        "height": 480,
    }
    args.update(overrides)
    return ImageAsset(**args)  # type: ignore[arg-type]


class TestObjectKey:
    def test_path_is_tenant_prefixed_and_sharded(self) -> None:
        key = ObjectKey("t1", ObjectKind.IMAGE_ASSET, SHA)
        assert key.path == f"tenants/t1/assets/{SHA[:2]}/{SHA}"

    @pytest.mark.parametrize("tenant", ["../x", "a/b", "", "a b", "x" * 65])
    def test_hostile_tenants_cannot_form_a_key(self, tenant: str) -> None:
        with pytest.raises(ContractError):
            ObjectKey(tenant, ObjectKind.RAW_DOCUMENT, SHA)

    @pytest.mark.parametrize("sha", ["", "../../etc/passwd", "A" * 64, "a" * 63, "g" * 64])
    def test_hostile_hashes_cannot_form_a_key(self, sha: str) -> None:
        with pytest.raises(ContractError):
            ObjectKey("t1", ObjectKind.RAW_DOCUMENT, sha)

    def test_stored_object_metadata_is_validated(self) -> None:
        key = ObjectKey("t1", ObjectKind.RAW_DOCUMENT, SHA)
        with pytest.raises(ContractError):
            StoredObject(key, -1, "image/png")
        with pytest.raises(ContractError):
            StoredObject(key, 1, "not a type")


class TestImageAsset:
    def test_valid_asset_and_search_text(self) -> None:
        a = make_asset(
            caption="Figure 3",
            alt_text=" torque map ",
            ocr_text="40 Nm",
            description="A diagram of bolt order",
            description_model_version="gpt-x@v1",
            bbox=(0.1, 0.2, 0.9, 0.8),
            page_number=4,
            nearby_chunk_ordinal=7,
        )
        assert a.search_text == "Figure 3\ntorque map\n40 Nm\nA diagram of bolt order"

    def test_no_text_means_empty_search_text(self) -> None:
        assert make_asset().search_text == ""

    @pytest.mark.parametrize(
        "change",
        [
            {"mime_type": "application/pdf"},
            {"mime_type": "image/"},
            {"byte_size": 0},
            {"byte_size": 51 * 1024 * 1024},
            {"width": 0},
            {"height": 20_001},
            {"page_number": 0},
            {"ordinal": -1},
            {"document_version": 0},
            {"bbox": (0.5, 0.2, 0.4, 0.8)},
            {"bbox": (0.0, 0.0, 1.5, 1.0)},
            {"caption": "x" * 5001},
            {"ocr_text": "a\x00b"},
            {"sha256": "abc"},
            {"tenant_id": "../x"},
            {"document_id": "nope"},
            {"description": "d"},  # model text without provenance
        ],
    )
    def test_invalid_assets_are_rejected(self, change: dict[str, object]) -> None:
        with pytest.raises(ContractError):
            make_asset(**change)


class TestChunkKind:
    def _args(self) -> dict[str, object]:
        return {
            "chunk_id": chunk_id_for(DOC, 1, 0),
            "document_id": DOC,
            "tenant_id": "t1",
            "document_version": 1,
            "ordinal": 0,
            "content": "a diagram",
            "content_hash": sha256_hex("a diagram"),
        }

    def test_image_chunk_needs_an_asset_and_text_chunk_must_not_have_one(self) -> None:
        image = Chunk(**self._args(), kind=ChunkKind.IMAGE, asset_sha256=SHA)  # type: ignore[arg-type]
        assert image.asset_sha256 == SHA
        with pytest.raises(ContractError):
            Chunk(**self._args(), kind=ChunkKind.IMAGE)  # type: ignore[arg-type]
        with pytest.raises(ContractError):
            Chunk(**self._args(), asset_sha256=SHA)  # type: ignore[arg-type]
        with pytest.raises(ContractError):
            replace(image, asset_sha256="not-a-hash")
