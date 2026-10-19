from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime

import pytest
from ekip_contracts.documents import (
    Chunk,
    ChunkEmbedding,
    ContractError,
    Document,
    DocumentAcl,
    DocumentStatus,
    chunk_id_for,
    document_id_for,
    sha256_hex,
)

NOW = datetime(2026, 10, 19, 6, 30, tzinfo=UTC)
ACL = DocumentAcl("acl-1", allowed_groups=frozenset({"engineering"}))


def make_doc(**overrides: object) -> Document:
    args: dict[str, object] = {
        "tenant_id": "tenant-dev",
        "source": "sharepoint",
        "source_document_id": "site/docs/manual.pdf",
        "version": 1,
        "source_version": "v7",
        "content": b"torque the bolts to 40 Nm",
        "title": "Manual",
        "mime_type": "application/pdf",
        "source_uri": "https://example.test/manual.pdf",
        "acl": ACL,
        "last_modified": NOW,
    }
    args.update(overrides)
    return Document.create(**args)  # type: ignore[arg-type]


class TestIdentity:
    def test_document_id_is_deterministic_and_prefixed(self) -> None:
        a = document_id_for("t1", "sharepoint", "x")
        assert a == document_id_for("t1", "sharepoint", "x")
        assert a.startswith("doc_") and len(a) == 36

    def test_tenants_never_collide_on_the_same_source_document(self) -> None:
        assert document_id_for("t1", "s", "x") != document_id_for("t2", "s", "x")

    def test_field_boundaries_are_unambiguous(self) -> None:
        assert document_id_for("t", "ab", "c") != document_id_for("t", "a", "bc")

    @pytest.mark.parametrize("tenant", ["", "../x", "a b", "x" * 65])
    def test_bad_tenant_is_rejected(self, tenant: str) -> None:
        with pytest.raises(ContractError):
            document_id_for(tenant, "s", "x")

    @pytest.mark.parametrize("source", ["", "SharePoint", "9x", "a-b"])
    def test_bad_source_is_rejected(self, source: str) -> None:
        with pytest.raises(ContractError):
            document_id_for("t", source, "x")

    @pytest.mark.parametrize("sid", ["", "   ", "a\x00b", "x" * 513])
    def test_bad_source_document_id_is_rejected(self, sid: str) -> None:
        with pytest.raises(ContractError):
            document_id_for("t", "s", sid)


class TestDocument:
    def test_create_derives_id_and_hash(self) -> None:
        d = make_doc()
        assert d.document_id == document_id_for("tenant-dev", "sharepoint", "site/docs/manual.pdf")
        assert d.content_hash == sha256_hex(b"torque the bolts to 40 Nm")
        assert d.status is DocumentStatus.PENDING and not d.is_searchable

    def test_same_content_same_hash_different_content_different_hash(self) -> None:
        assert make_doc().content_hash == make_doc(version=2).content_hash
        assert make_doc().content_hash != make_doc(content=b"other").content_hash

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            make_doc().version = 2  # type: ignore[misc]

    def test_only_indexed_is_searchable(self) -> None:
        for status in DocumentStatus:
            assert replace(make_doc(), status=status).is_searchable == (
                status is DocumentStatus.INDEXED
            )

    def test_forged_document_id_is_rejected(self) -> None:
        d = make_doc()
        with pytest.raises(ContractError, match="document_id"):
            replace(d, document_id=document_id_for("tenant-other", "sharepoint", "x"))

    @pytest.mark.parametrize(
        "change",
        [
            {"version": 0},
            {"source_version": " "},
            {"title": ""},
            {"mime_type": "pdf"},
            {"mime_type": "Application/PDF"},
            {"source_uri": ""},
            {"last_modified": datetime(2026, 1, 1)},  # naive
        ],
    )
    def test_invalid_fields_are_rejected(self, change: dict[str, object]) -> None:
        with pytest.raises(ContractError):
            make_doc(**change)

    def test_bad_content_hash_is_rejected(self) -> None:
        with pytest.raises(ContractError, match="sha256"):
            replace(make_doc(), content_hash="abc")


class TestAcl:
    def test_empty_acl_grants_nobody(self) -> None:
        acl = DocumentAcl("v1")
        assert not acl.grants_anyone
        assert not acl.permits("alice", frozenset({"engineering"}))

    def test_user_or_group_match(self) -> None:
        acl = DocumentAcl("v1", frozenset({"alice"}), frozenset({"finance"}))
        assert acl.permits("alice", frozenset())
        assert acl.permits("bob", frozenset({"finance"}))
        assert not acl.permits("bob", frozenset({"engineering"}))

    def test_group_names_are_matched_exactly(self) -> None:
        acl = DocumentAcl("v1", allowed_groups=frozenset({"finance"}))
        assert not acl.permits("x", frozenset({"Finance", "finance-eu", "fin"}))

    @pytest.mark.parametrize("bad", ["", "a b", "x\n", "a" * 257])
    def test_bad_principals_are_rejected(self, bad: str) -> None:
        with pytest.raises(ContractError):
            DocumentAcl("v1", allowed_users=frozenset({bad}))

    def test_bad_acl_version_is_rejected(self) -> None:
        with pytest.raises(ContractError):
            DocumentAcl("")


class TestChunk:
    def test_ids_hash_and_tenant_come_from_the_document(self) -> None:
        d = make_doc(version=7)
        c = Chunk.create(
            document=d,
            ordinal=8,
            content="hello",
            section_path=("Ch 3", "Maintenance"),
            page_number=42,
        )
        assert c.chunk_id == f"{d.document_id}:v7:c0008"
        assert c.tenant_id == "tenant-dev" and c.document_version == 7
        assert c.content_hash == sha256_hex("hello")

    def test_same_text_in_two_versions_shares_a_hash_but_not_an_id(self) -> None:
        a = Chunk.create(document=make_doc(version=1), ordinal=0, content="same")
        b = Chunk.create(document=make_doc(version=2), ordinal=0, content="same")
        assert a.content_hash == b.content_hash and a.chunk_id != b.chunk_id

    @pytest.mark.parametrize("content", ["", "   ", "a\x00b", "x" * 20_001])
    def test_bad_content_is_rejected(self, content: str) -> None:
        with pytest.raises(ContractError):
            Chunk.create(document=make_doc(), ordinal=0, content=content)

    def test_tampered_content_or_id_is_rejected(self) -> None:
        c = Chunk.create(document=make_doc(), ordinal=0, content="hello")
        with pytest.raises(ContractError, match="content_hash"):
            replace(c, content="changed")
        with pytest.raises(ContractError, match="chunk_id"):
            replace(c, ordinal=1)

    def test_section_and_page_limits(self) -> None:
        d = make_doc()
        with pytest.raises(ContractError):
            Chunk.create(document=d, ordinal=0, content="x", section_path=("a",) * 13)
        with pytest.raises(ContractError):
            Chunk.create(document=d, ordinal=0, content="x", page_number=0)

    def test_chunk_id_for_validates(self) -> None:
        with pytest.raises(ContractError):
            chunk_id_for("not-a-doc-id", 1, 0)
        with pytest.raises(ContractError):
            chunk_id_for(make_doc().document_id, 0, 0)


class TestEmbedding:
    def _chunk(self) -> Chunk:
        return Chunk.create(document=make_doc(), ordinal=0, content="hello")

    def test_valid_embedding(self) -> None:
        c = self._chunk()
        e = ChunkEmbedding(c.chunk_id, c.content_hash, "gemini-embedding-001@768", (0.1, -0.2))
        assert e.dimensions == 2

    @pytest.mark.parametrize("vector", [(), (float("nan"),), (float("inf"),), (0.0,) * 8193])
    def test_bad_vectors_are_rejected(self, vector: tuple[float, ...]) -> None:
        c = self._chunk()
        with pytest.raises(ContractError):
            ChunkEmbedding(c.chunk_id, c.content_hash, "m1", vector)

    def test_bad_ids_are_rejected(self) -> None:
        c = self._chunk()
        with pytest.raises(ContractError):
            ChunkEmbedding("nope", c.content_hash, "m1", (0.1,))
        with pytest.raises(ContractError):
            ChunkEmbedding(c.chunk_id, "xyz", "m1", (0.1,))
        with pytest.raises(ContractError):
            ChunkEmbedding(c.chunk_id, c.content_hash, "bad model!", (0.1,))
