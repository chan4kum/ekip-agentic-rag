import io

import pytest
from doc_factory import docx, pdf_with_text, png, scanned_pdf
from ekip_contracts.parsing import BlockKind, ParseError
from ekip_ingestion import default_registry
from ekip_ingestion.images import inspect_image, is_meaningful
from ekip_ingestion.limits import DEFAULT_LIMITS, ParseLimits
from ekip_ingestion.parsers.docx import DOCX_MIME, DocxParser
from ekip_ingestion.parsers.pdf import PdfParser
from pypdf import PdfWriter

K = BlockKind


class TestPdf:
    async def test_text_pages_keep_their_page_numbers_and_need_no_ocr(self) -> None:
        data = pdf_with_text([["Torque the bolts to 40 Nm."], ["Second page text."]])
        doc = await PdfParser().parse(data, "application/pdf")
        by_page = {b.page_number: b.text for b in doc.blocks}
        assert "40 Nm" in by_page[1] and "Second page" in by_page[2]
        assert doc.pages_needing_ocr == () and all(b.kind is K.PARAGRAPH for b in doc.blocks)

    async def test_a_blank_page_is_flagged_for_ocr_only_where_needed(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(612, 792)
        buf = io.BytesIO()
        writer.write(buf)
        doc = await PdfParser().parse(buf.getvalue(), "application/pdf")
        assert doc.pages_needing_ocr == (1,) and doc.blocks == ()
        assert doc.warnings == ("1 page(s) without a text layer",)

    async def test_scanned_pdf_gives_images_and_flags_ocr(self) -> None:
        pic = png(300, 200)
        doc = await PdfParser().parse(scanned_pdf([pic]), "application/pdf")
        images = [b for b in doc.blocks if b.kind is K.IMAGE]
        assert doc.pages_needing_ocr == (1,) and len(images) == 1
        assert images[0].page_number == 1 and images[0].image is not None
        assert images[0].image.width > 0

    async def test_an_image_repeated_on_every_page_is_kept_once(self) -> None:
        logo = png(300, 200)
        doc = await PdfParser().parse(scanned_pdf([logo, logo, logo]), "application/pdf")
        assert len([b for b in doc.blocks if b.kind is K.IMAGE]) == 1
        assert any("repeated image" in w for w in doc.warnings)

    async def test_encrypted_pdf_is_refused(self) -> None:
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.encrypt("secret")
        buf = io.BytesIO()
        writer.write(buf)
        with pytest.raises(ParseError, match="encrypted"):
            await PdfParser().parse(buf.getvalue(), "application/pdf")

    @pytest.mark.parametrize("data", [b"%PDF-1.4 garbage garbage", b"%PDF-" + bytes(100)])
    async def test_corrupt_pdf_is_a_parse_error_not_a_crash(self, data: bytes) -> None:
        with pytest.raises(ParseError):
            await PdfParser().parse(data, "application/pdf")

    async def test_wrong_magic_bytes_and_limits(self) -> None:
        with pytest.raises(ParseError, match="not a PDF"):
            await PdfParser().parse(b"MZ\x90\x00 definitely an exe", "application/pdf")
        too_many = pdf_with_text([["a"], ["b"], ["c"]])
        with pytest.raises(ParseError, match="too many pages"):
            await PdfParser(ParseLimits(max_pages=2)).parse(too_many, "application/pdf")


class TestDocx:
    async def test_structure_styles_table_and_title(self) -> None:
        doc = await DocxParser().parse(docx(), DOCX_MIME)
        assert doc.title == "Maintenance Manual"
        got = [(b.kind, b.text, b.level) for b in doc.blocks]
        assert (K.HEADING, "Chapter 1", 1) in got and (K.HEADING, "Torque values", 2) in got
        assert (K.PARAGRAPH, "Tighten the bolts in a star pattern.", None) in got
        assert (K.LIST_ITEM, "Check the gasket", None) in got
        assert (K.TABLE, "| Bolt | Torque |\n| --- | --- |\n| M8 | 25 Nm |", None) in got

    async def test_embedded_picture_is_extracted(self) -> None:
        pic = png(300, 200)
        doc = await DocxParser().parse(docx(picture=pic), DOCX_MIME)
        images = [b for b in doc.blocks if b.kind is K.IMAGE]
        assert len(images) == 1 and images[0].image is not None
        assert (images[0].image.width, images[0].image.height) == (300, 200)

    async def test_tiny_pictures_are_dropped(self) -> None:
        doc = await DocxParser().parse(docx(picture=png(10, 10)), DOCX_MIME)
        assert not [b for b in doc.blocks if b.kind is K.IMAGE]

    async def test_zip_bomb_is_rejected_before_it_is_opened(self) -> None:
        data = docx(extra_bytes=5_000)
        with pytest.raises(ParseError, match="expands beyond"):
            await DocxParser(ParseLimits(max_zip_uncompressed_bytes=1_000)).parse(data, DOCX_MIME)

    @pytest.mark.parametrize("data", [b"not a zip at all", b"PK\x03\x04 truncated"])
    async def test_not_a_docx(self, data: bytes) -> None:
        with pytest.raises(ParseError):
            await DocxParser().parse(data, DOCX_MIME)

    async def test_a_zip_that_is_not_a_docx_is_a_parse_error(self) -> None:
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("hello.txt", "hi")
        with pytest.raises(ParseError):
            await DocxParser().parse(buf.getvalue(), DOCX_MIME)


class TestImages:
    def test_png_metadata(self) -> None:
        info = inspect_image(png(120, 80), DEFAULT_LIMITS)
        assert info is not None and (info.mime_type, info.width, info.height) == (
            "image/png",
            120,
            80,
        )

    def test_svg_garbage_and_oversized_are_rejected(self) -> None:
        assert inspect_image(b"<svg xmlns='http://www.w3.org/2000/svg'/>", DEFAULT_LIMITS) is None
        assert inspect_image(b"nope", DEFAULT_LIMITS) is None
        assert inspect_image(b"", DEFAULT_LIMITS) is None
        assert inspect_image(png(), ParseLimits(max_image_bytes=100)) is None

    def test_decompression_bomb_is_rejected_by_declared_pixels(self) -> None:
        assert inspect_image(png(200, 200), ParseLimits(max_image_pixels=10_000)) is None

    def test_meaningfulness_filter(self) -> None:
        data = png(200, 200)
        info = inspect_image(data, DEFAULT_LIMITS)
        assert info is not None and is_meaningful(info, len(data), DEFAULT_LIMITS)
        small = inspect_image(png(20, 20), DEFAULT_LIMITS)
        assert small is not None and not is_meaningful(small, 5_000, DEFAULT_LIMITS)
        assert not is_meaningful(info, 100, DEFAULT_LIMITS)  # too few bytes (flat icon)


class TestRegistry:
    async def test_routes_by_mime_ignoring_parameters_and_case(self) -> None:
        reg = default_registry()
        doc = await reg.parse(b"hello", "Text/Plain; charset=utf-8")
        assert doc.blocks[0].text == "hello"

    async def test_unsupported_type(self) -> None:
        with pytest.raises(ParseError, match="unsupported"):
            await default_registry().parse(b"MZ", "application/x-msdownload")

    def test_supported_types(self) -> None:
        assert {"application/pdf", DOCX_MIME, "text/html", "text/markdown", "text/plain"} <= (
            default_registry().supported
        )

    async def test_declared_type_cannot_smuggle_a_binary_into_the_wrong_parser(self) -> None:
        with pytest.raises(ParseError):
            await default_registry().parse(docx(), "application/pdf")  # a DOCX claiming to be a PDF
