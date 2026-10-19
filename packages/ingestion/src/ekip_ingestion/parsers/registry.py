"""Pick a parser by MIME type. The declared type is a hint: each parser also checks magic bytes."""

from ekip_contracts.parsing import DocumentParser, ParsedDocument, ParseError

from ekip_ingestion.limits import DEFAULT_LIMITS, ParseLimits
from ekip_ingestion.parsers.docx import DocxParser
from ekip_ingestion.parsers.html import HtmlParser
from ekip_ingestion.parsers.pdf import PdfParser
from ekip_ingestion.parsers.text import TextParser


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._by_type: dict[str, DocumentParser] = {}
        for parser in parsers:
            for mime in parser.supported_mime_types:
                self._by_type[mime] = parser

    @property
    def supported(self) -> frozenset[str]:
        return frozenset(self._by_type)

    async def parse(self, data: bytes, mime_type: str) -> ParsedDocument:
        mime = mime_type.split(";", 1)[0].strip().lower()
        parser = self._by_type.get(mime)
        if parser is None:
            raise ParseError("unsupported document type")
        return await parser.parse(data, mime)


def default_registry(limits: ParseLimits = DEFAULT_LIMITS) -> ParserRegistry:
    return ParserRegistry(
        [TextParser(limits), HtmlParser(limits), PdfParser(limits), DocxParser(limits)]
    )
