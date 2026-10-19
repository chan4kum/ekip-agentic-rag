import random
from datetime import UTC, datetime

import pytest
from doc_factory import docx, png
from ekip_contracts.documents import (
    Chunk,
    ChunkKind,
    ContractError,
    Document,
    DocumentAcl,
    sha256_hex,
)
from ekip_contracts.parsing import BlockKind, ParsedBlock, ParsedDocument, ParsedImage
from ekip_ingestion.chunking import (
    ChunkingConfig,
    ImageEnrichment,
    chunk_document,
    embedding_text,
    strip_boilerplate,
)
from ekip_ingestion.parsers.docx import DOCX_MIME, DocxParser

K = BlockKind
CFG = ChunkingConfig(target_chars=300, max_chars=500, overlap_chars=60)


def doc(version: int = 1) -> Document:
    return Document.create(
        tenant_id="t1",
        source="sharepoint",
        source_document_id="x",
        version=version,
        source_version="r1",
        content=b"raw",
        title="Manual",
        mime_type="text/plain",
        source_uri="https://example.test/x",
        acl=DocumentAcl("a1", frozenset({"u"})),
        last_modified=datetime(2026, 10, 19, tzinfo=UTC),
    )


def para(text: str, page: int | None = None) -> ParsedBlock:
    return ParsedBlock(K.PARAGRAPH, text, page_number=page)


def head(text: str, level: int) -> ParsedBlock:
    return ParsedBlock(K.HEADING, text, level=level)


def parsed(*blocks: ParsedBlock) -> ParsedDocument:
    return ParsedDocument(tuple(blocks))


def run(*blocks: ParsedBlock, cfg: ChunkingConfig = CFG, **kw):  # type: ignore[no-untyped-def]
    return chunk_document(doc(), parsed(*blocks), cfg, **kw)


def sentence(i: int) -> str:
    return f"Sentence number {i} explains the torque procedure in detail."


class TestSections:
    def test_chunks_never_span_headings_and_carry_the_section_path(self) -> None:
        r = run(
            head("Ch 1", 1),
            para("intro"),
            head("Torque", 2),
            para("a"),
            head("Safety", 2),
            para("b"),
            head("Ch 2", 1),
            para("c"),
        )
        got = [(c.section_path, c.content) for c in r.chunks]
        assert got == [
            (("Ch 1",), "intro"),
            (("Ch 1", "Torque"), "a"),
            (("Ch 1", "Safety"), "b"),
            (("Ch 2",), "c"),
        ]

    def test_text_before_any_heading_has_an_empty_path(self) -> None:
        assert run(para("preamble"), head("A", 1), para("x")).chunks[0].section_path == ()

    def test_heading_only_sections_produce_no_empty_chunks(self) -> None:
        r = run(head("A", 1), head("B", 2), para("x"))
        assert [c.content for c in r.chunks] == ["x"] and r.chunks[0].section_path == ("A", "B")

    def test_very_deep_or_long_headings_are_capped(self) -> None:
        blocks = [head("h" * 400, 1)] + [head(f"s{i}", 2 + i % 4) for i in range(5)]
        r = run(*blocks, para("x"))
        assert all(len(p) <= 256 for p in r.chunks[0].section_path)


class TestPacking:
    def test_small_paragraphs_are_packed_up_to_the_target(self) -> None:
        r = run(*[para(sentence(i)) for i in range(20)])
        assert len(r.chunks) > 1
        assert all(len(c.content) <= CFG.target_chars for c in r.chunks)
        assert [c.ordinal for c in r.chunks] == list(range(len(r.chunks)))

    def test_no_text_is_lost_or_duplicated_when_packing(self) -> None:
        texts = [sentence(i) for i in range(30)]
        r = run(*[para(t) for t in texts])
        joined = "\n\n".join(c.content for c in r.chunks)
        assert [t for t in texts if t in joined] == texts and joined.count(
            "Sentence number 7 "
        ) == 1

    def test_list_items_stay_together_on_separate_lines(self) -> None:
        r = run(ParsedBlock(K.LIST_ITEM, "one"), ParsedBlock(K.LIST_ITEM, "two"))
        assert r.chunks[0].content == "one\ntwo"

    def test_page_number_is_that_of_the_first_block(self) -> None:
        r = run(para("a", 3), para("b", 4))
        assert len(r.chunks) == 1 and r.chunks[0].page_number == 3


class TestOversize:
    def test_long_paragraph_splits_at_sentences_with_overlap_and_loses_nothing(self) -> None:
        sentences = [sentence(i) for i in range(30)]
        r = run(para(" ".join(sentences)))
        assert len(r.chunks) > 2 and all(len(c.content) <= CFG.max_chars for c in r.chunks)
        for earlier, later in zip(r.chunks, r.chunks[1:], strict=False):
            tail = earlier.content.split(". ")[-1]
            assert tail[:20] in later.content  # the overlap repeats the previous tail
        assert all(any(s in c.content for c in r.chunks) for s in sentences)

    def test_text_without_any_punctuation_is_cut_at_whitespace_within_the_ceiling(self) -> None:
        words = " ".join(f"word{i}" for i in range(400))
        r = run(para(words))
        assert all(len(c.content) <= CFG.max_chars for c in r.chunks)
        rebuilt = " ".join(c.content for c in r.chunks)
        assert all(f"word{i} " in rebuilt + " " for i in range(400))

    def test_one_unbroken_token_is_hard_cut(self) -> None:
        r = run(para("x" * 1_700))
        assert len(r.chunks) >= 4 and all(len(c.content) <= CFG.max_chars for c in r.chunks)
        assert sum(len(c.content) for c in r.chunks) >= 1_700

    def test_big_table_splits_by_rows_repeating_the_header(self) -> None:
        rows = [f"| M{i} | {i} Nm |" for i in range(100)]
        table = "\n".join(["| Bolt | Torque |", "| --- | --- |", *rows])
        r = run(ParsedBlock(K.TABLE, table))
        assert len(r.chunks) > 1
        for c in r.chunks:
            assert c.content.startswith("| Bolt | Torque |\n| --- | --- |")
            assert len(c.content) <= CFG.max_chars
        body = "\n".join(c.content for c in r.chunks)
        assert all(row in body for row in rows)

    def test_small_table_and_code_are_their_own_chunks(self) -> None:
        r = run(
            para("before"),
            ParsedBlock(K.TABLE, "| a |\n| --- |\n| 1 |"),
            ParsedBlock(K.CODE, "x = 1"),
            para("after"),
        )
        assert [c.content for c in r.chunks] == [
            "before",
            "| a |\n| --- |\n| 1 |",
            "x = 1",
            "after",
        ]

    def test_long_code_splits_on_lines_and_an_endless_line_is_cut(self) -> None:
        code = "\n".join(f"line_{i} = {i}" for i in range(200))
        r = run(ParsedBlock(K.CODE, code))
        assert len(r.chunks) > 1 and all(len(c.content) <= CFG.max_chars for c in r.chunks)
        assert "line_199 = 199" in r.chunks[-1].content
        r2 = run(ParsedBlock(K.CODE, "y" * 1_200))
        assert all(len(c.content) <= CFG.max_chars for c in r2.chunks)


class TestDeterminismAndStability:
    def blocks(self) -> list[ParsedBlock]:
        out: list[ParsedBlock] = []
        for s in range(4):
            out.append(head(f"Section {s}", 1))
            out += [para(sentence(s * 10 + i)) for i in range(8)]
        return out

    def test_same_input_gives_identical_ids_and_hashes(self) -> None:
        assert run(*self.blocks()).chunks == run(*self.blocks()).chunks

    def test_editing_the_last_section_leaves_earlier_chunks_untouched(self) -> None:
        a = run(*self.blocks()).chunks
        edited = self.blocks()
        edited[-1] = para("A completely different final paragraph.")
        b = run(*edited).chunks
        stable = [x.content_hash for x in a if x.section_path != ("Section 3",)]
        assert stable and stable == [x.content_hash for x in b if x.section_path != ("Section 3",)]

    def test_editing_one_section_does_not_shift_other_sections_chunks(self) -> None:
        """Because sections are hard boundaries, an edit early on only changes ITS section."""
        a = run(*self.blocks()).chunks
        edited = self.blocks()
        edited[1] = para("Rewritten first paragraph of section zero, much longer " * 3)
        b = run(*edited).chunks

        def others(cs: tuple[Chunk, ...]) -> list[str]:
            return [c.content_hash for c in cs if c.section_path != ("Section 0",)]

        assert others(a) == others(b)

    def test_a_new_document_version_changes_ids_but_not_content_hashes(self) -> None:
        r1 = chunk_document(doc(1), parsed(para("same text")), CFG)
        r2 = chunk_document(doc(2), parsed(para("same text")), CFG)
        assert r1.chunks[0].chunk_id != r2.chunks[0].chunk_id
        assert r1.chunks[0].content_hash == r2.chunks[0].content_hash

    def test_random_documents_always_yield_valid_bounded_chunks(self) -> None:
        rng = random.Random(7)  # noqa: S311 - deterministic test data, not security
        for _ in range(40):
            blocks: list[ParsedBlock] = []
            for _ in range(rng.randint(1, 25)):
                kind = rng.choice(["p", "p", "h", "t", "c", "l"])
                n = rng.choice([5, 40, 400, 900, 2500])
                text = " ".join(f"w{rng.randint(0, 9999)}." for _ in range(n // 6 + 1))
                blocks.append(
                    head(text[:80], rng.randint(1, 4))
                    if kind == "h"
                    else ParsedBlock(K.TABLE, f"| h |\n| --- |\n| {text} |")
                    if kind == "t"
                    else ParsedBlock(K.CODE, text)
                    if kind == "c"
                    else ParsedBlock(K.LIST_ITEM, text)
                    if kind == "l"
                    else para(text)
                )
            r = run(*blocks)
            assert all(0 < len(c.content) <= CFG.max_chars for c in r.chunks)


class TestImages:
    def image(self, caption: str | None = None, alt: str | None = None) -> ParsedBlock:
        data = png(200, 150)
        return ParsedBlock(
            K.IMAGE,
            text=caption or alt or "",
            page_number=2,
            image=ParsedImage(data, "image/png", 200, 150, caption=caption, alt_text=alt),
        )

    def test_captioned_image_becomes_an_asset_and_an_image_chunk_in_order(self) -> None:
        img = self.image(caption="Figure 3: bolt order", alt="diagram")
        r = run(para("before"), img, para("after"))
        assert [c.kind for c in r.chunks] == [ChunkKind.TEXT, ChunkKind.IMAGE, ChunkKind.TEXT]
        chunk, asset = r.chunks[1], r.assets[0]
        assert chunk.asset_sha256 == asset.asset.sha256 == sha256_hex(asset.data)
        assert chunk.content == "Figure 3: bolt order\ndiagram" and chunk.page_number == 2
        assert asset.asset.nearby_chunk_ordinal == 0 and asset.asset.page_number == 2
        assert asset.asset.ordinal == 0 and asset.asset.byte_size == len(asset.data)

    def test_image_without_any_text_is_stored_but_not_searchable_yet(self) -> None:
        r = run(para("x"), self.image())
        assert len(r.assets) == 1 and [c.kind for c in r.chunks] == [ChunkKind.TEXT]

    def test_enrichment_from_ocr_or_vision_makes_it_searchable_with_provenance(self) -> None:
        img = self.image()
        sha = sha256_hex(img.image.data) if img.image else ""
        enrich = {sha: ImageEnrichment("40 Nm", "A bolt pattern diagram", "vision-x@v1")}
        r = run(para("x"), img, enrichments=enrich)
        chunk = r.chunks[1]
        assert chunk.kind is ChunkKind.IMAGE and chunk.content == "40 Nm\nA bolt pattern diagram"
        a = r.assets[0].asset
        assert (a.ocr_text, a.description, a.description_model_version) == (
            "40 Nm",
            "A bolt pattern diagram",
            "vision-x@v1",
        )

    def test_enrichment_for_other_images_is_ignored(self) -> None:
        r = run(self.image(), enrichments={sha256_hex(b"other"): ImageEnrichment("x", "y", "m@1")})
        assert r.assets[0].asset.description is None

    def test_description_without_a_model_version_is_refused(self) -> None:
        img = self.image()
        sha = sha256_hex(img.image.data) if img.image else ""
        with pytest.raises(ContractError):
            run(img, enrichments={sha: ImageEnrichment(None, "text", None)})

    def test_asset_ordinals_follow_document_order(self) -> None:
        r = run(self.image(caption="a"), para("x"), self.image(caption="b"))
        assert [p.asset.ordinal for p in r.assets] == [0, 1]


class TestBoilerplate:
    def test_running_headers_footers_and_page_numbers_are_removed(self) -> None:
        blocks: list[ParsedBlock] = []
        for p in range(1, 7):
            blocks += [
                para("ACME Maintenance Manual", p),
                para(f"Unique content of page {p}.", p),
                para(str(p), p),
                para(f"Page {p} of 6", p),
            ]
        out = strip_boilerplate(parsed(*blocks))
        assert [b.text for b in out.blocks] == [f"Unique content of page {p}." for p in range(1, 7)]

    def test_short_repeats_below_the_threshold_are_kept(self) -> None:
        out = strip_boilerplate(parsed(para("Note", 1), para("Note", 2), para("body", 3)))
        assert [b.text for b in out.blocks] == ["Note", "Note", "body"]

    def test_text_without_page_numbers_is_never_treated_as_furniture(self) -> None:
        blocks = [para("12") for _ in range(5)]
        assert len(strip_boilerplate(parsed(*blocks)).blocks) == 5


def test_embedding_text_adds_title_and_path_but_not_to_the_stored_chunk() -> None:
    r = run(head("Ch 1", 1), head("Torque", 2), para("Use 40 Nm."))
    c = r.chunks[0]
    assert embedding_text(doc(), c) == "Manual > Ch 1 > Torque\nUse 40 Nm."
    assert c.content == "Use 40 Nm."
    assert embedding_text(doc(), run(para("plain")).chunks[0]) == "Manual\nplain"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_chars": 100},
        {"target_chars": 900, "max_chars": 800},
        {"max_chars": 30_000},
        {"overlap_chars": 1_000},
        {"overlap_chars": -1},
    ],
)
def test_bad_config_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ContractError):
        ChunkingConfig(**kwargs)


async def test_parse_then_chunk_a_real_docx_end_to_end() -> None:
    parsed_doc = await DocxParser().parse(docx(picture=png(300, 200)), DOCX_MIME)
    r = chunk_document(doc(), parsed_doc)
    texts = " ".join(c.content for c in r.chunks)
    assert "star pattern" in texts and "M8" in texts and len(r.assets) == 1
    assert any(c.section_path == ("Chapter 1", "Torque values") for c in r.chunks)
